"""
Main script for running simulations.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from context import knockpy, mlr_src
from mlr_src import gen_data, oracle, parser, utilities
from mlr_src.utilities import elapsed
from knockpy import knockoffs
from knockpy.knockoff_filter import KnockoffFilter as KF
from knockpy import knockoff_stats as kstats

# bayes coeff diff, requires pyblip
try:
	from mlr_src import bcd
	bcd_available = True
except ImportError as e:
	bcd_available = False
	warnings.warn(f"Cannot import bcd, raised error {e}")

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

# Output
columns = [
	"n",
	"p",
	"sparsity",
	"coeff_size",
	"seed",
	"cond_mean",
	"mx",
	"knockoff_type",
	"fstat",
	"q",
	"power",
	"fdp",
	"fstat_time",
	"ko_time",
]

def single_seed_sim(
	seed,
	n,
	p,
	covmethod,
	mx,
	t0,
	args
):
	print(f"At seed={seed}, n={n}, p={p}.")

	# 1. Create data-generating process for X
	output = []
	np.random.seed(seed)
	dgprocess = knockpy.dgp.DGP()
	max_corr = args.get('max_corr', [0.99])[0]
	if covmethod not in ['orthogonal', 'ark']:
		sample_kwargs = {}
		if covmethod in ['ver', 'ar1']:
			sample_kwargs['max_corr'] = max_corr
		dgprocess.sample_data(
			n=n,
			p=p,
			method=covmethod,
			**sample_kwargs,
			a=args.get("a", [5])[0],
			b=args.get("b", [1])[0],
		)
		X = dgprocess.X
		Sigma = dgprocess.Sigma
	elif covmethod == 'ark':
		X, Sigma = gen_data.gen_ark_X(
			n=n, p=p, 
			k=args.get("k", [2])[0],
			max_corr=max_corr,  
		)
	elif covmethod == 'orthogonal':
		X = stats.ortho_group.rvs(n)
		X = X[:, 0:p]
		Sigma = np.eye(p)

	# 2. Create knockoffs
	S_methods = args.get("s_method", ['mvr', 'sdp'])
	for S_method in S_methods:
		time0 = time.time()
		if mx:
			ksampler = knockoffs.GaussianSampler(
				X=X, Sigma=Sigma, method=S_method
			)
		else:
			ksampler = knockoffs.FXSampler(X=X, method=S_method)
		S = ksampler.fetch_S()
		S_time = np.around(time.time() - time0, 2)
		print(f"Finished with {S_method} S-matrix comp for seed={seed}, took {S_time} at {elapsed(t0)}.")
		Xk = ksampler.sample_knockoffs()
		ko_time = np.around(time.time() - time0, 2)
		print(f"Finished sampling {S_method} knockoffs for seed={seed}, took {ko_time} at {elapsed(t0)}.")

		# 3. Generate y
		for sparse in args.get('sparsity', [0.1]):
			for coeff_dist in args.get('coeff_dist', ['uniform']):
				for coeff_size in args.get('coeff_size', [1]):
					# options: gaussian, expo, probit, logistic
					for y_dist in args.get("y_dist", ['gaussian']): 
						for cond_mean in args.get("cond_mean", ['linear']):
							# Sample beta
							beta = gen_data.sample_beta(
								p=p,
								sparsity=sparse,
								coeff_dist=coeff_dist,
								coeff_size=coeff_size,
							)
							# Sample y
							if cond_mean == 'linear':
								mu = np.dot(X, beta)
							elif cond_mean == 'cos':
								mu = np.dot(np.cos(X), beta)
							else:
								raise ValueError(f"unrecognized cond_mean={cond_mean}")
							y = gen_data.sample_y(mu=mu, y_dist=y_dist)

							# Assemble feature statistics
							fstats = []
							# oracle
							if not mx:
								fstats.append((oracle.OracleFXStatistic(beta=beta), 'oracle'))
							# Linear statistics
							if args.get("compute_lcd", [True])[0]:
								fstats.append(('lcd', 'lcd'))
							if args.get('compute_lsm', [True])[0]:
								fstats.append(('lsm', 'lsm'))
							# MLR statistics + bayesian baseline
							if args.get('compute_mlr', [True])[0]:
								fstats.append(('mlr', 'mlr'))
							if args.get("compute_mlr_spline", [False])[0]:
								fstats.append(('mlr_spline', 'mlr_spline'))
							if args.get("compute_bcd", [False])[0]:
								fstats.append((mlr_src.bcd.BayesCoeffDiff(), 'bcd'))
							# nonparametric statistics
							if args.get("compute_deeppink", [False])[0]:
								fstats.append(('deeppink', 'deeppink'))
							if args.get("compute_randomforest", [False])[0]:
								fstats.append(('randomforest', 'randomforest'))

							for fstat, fstatname in fstats:
								# initialize KF
								kf = KF(ksampler=ksampler, fstat=fstat)
								fstat_kwargs = dict()
								if fstat in ['mlr', 'bcd']:
									fstat_kwargs['n_iter'] = args.get("n_iter", [2000])[0]
									fstat_kwargs['chains'] = args.get("chains", [5])[0]
								# Run KF
								time0 = time.time()
								kf.forward(
									X=X, Xk=Xk, y=y, fstat_kwargs=fstat_kwargs, fdr=0.1,
								)
								fstat_time = time.time() - time0
								for q in args.get("q", [0.05, 0.10, 0.15, 0.20]):
									T = kstats.data_dependent_threshhold(W=kf.W, fdr=q)
									rej = (kf.W >= T).astype("float32")
									power, fdp = utilities.calc_power_fdr(
										rej,
										beta,
									)

									# Add output
									output.append([
										n,
										p,
										sparse,
										coeff_size,
										cond_mean,
										seed,
										mx,
										S_method,
										fstatname,
										q,
										power,
										fdp,
										fstat_time,
										ko_time
									])
	return output



def main(args):

	# Parse and extract arguments
	t0 = time.time()
	args = parser.parse_args(sys.argv)

	# # 2. how to compute knockoffs 
	# q = args.get("q", [0.05])[0]
	# mx = args.get("mx", [False])[0]
	# S_methods = args.get("s_method", ['mvr', 'sdp'])
	# # 3. feature statistics
	# lambd_scalings = args.get('lambd', [64])
	# compute_mlr_fx = args.get("compute_mlr_fx", [True])[0]
	# compute_mlr_mx = args.get("compute_mlr_mx", [True])[0]
	# compute_bayes_baseline = args.get('compute_bayes_baseline', [True])[0]
	# compute_lcd = args.get("compute_lcd", [True])[0]
	# compute_lsm = args.get("compute_lsm", [True])[0]
	# n_iter = args.get("n_iter", [5000])[0]
	# chains = args.get("chains", [3])[0]
	# how_est_plugins = args.get("how_est", [''])#["xi", "gd_cvx", "gd_noncvx", "em"])
	# if len(how_est_plugins) == 1 and how_est_plugins[0] == '':
	# 	how_est_plugins = []

	# keywords for the data-generating process for X
	seed_start = args.get("seed_start", [0])[0]
	reps = args.get('reps', [1])[0] # number of replications
	num_processes = args.get('num_processes', [1])[0]
	ps = args.get('p', [300]) # dimensionality
	kappas = args.get('kappa', [2.1, 3, 4]) # ratio of n/p
	covmethods = args.get('covmethod', ['AR1']) # cov matrix

	# Save args, create output dir
	output_dir = utilities.create_output_directory(args, dir_type=DIR_TYPE)
	output_path = output_dir + 'results.csv'
	summary_path = output_dir + 'summary.csv'

	# Run and save output
	all_outputs = []
	for p in ps:
		for kappa in kappas:
			n = int(p * kappa)
			for covmethod in covmethods:
				for mx in args.get("mx", [True]):
					outputs = utilities.apply_pool(
						func=single_seed_sim,
						seed=list(range(seed_start+1, seed_start+reps+1)), 
						constant_inputs={
							'n':n,
							'p':p,
							'covmethod':covmethod,
							'mx':mx,
							't0':t0,
							'args':args,
						},
						num_processes=num_processes, 
					)
					for output in outputs:
						all_outputs.extend(output)

					# Turn into pandas dataframe
					output_df = pd.DataFrame(all_outputs, columns=columns)
					output_df.to_csv(output_path, index=False)
					summary = output_df.groupby(
						['mx', 'knockoff_type', 'fstat', 'n', 'p', 'sparsity', 'coeff_size', 'q'] # 'lambd' for plug-in ests
					)[[
						'power', 'fdp', 'fstat_time', 'ko_time',
					]].agg(
						["mean", "std"]
					).reset_index()
					summary.to_csv(summary_path, index=False)
					pd.set_option('display.max_rows', 500)
					#pd.set_option('display.max_columns', 10)
					print(summary)
					pd.reset_option('display.max_rows|display.max_columns|display.width')




if __name__ == '__main__':
	main(sys.argv)
