"""
Main script for running simulations.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from context import knockpy, mlr_src
from mlr_src import gen_data, oracle, parser, utilities
from mlr_src.utilities import elapsed
from mlr_src.studentized import StudentizedLassoStatistic
from knockpy import knockoffs
from knockpy.knockoff_filter import KnockoffFilter as KF
from knockpy import knockoff_stats as kstats
import warnings

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
	"covmethod",
	"sparsity",
	"coeff_size",
	"coeff_dist",
	"seed",
	"cond_mean",
	"y_dist",
	"mx",
	"knockoff_type",
	"fstat",
	"q",
	"power",
	"fdp",
	"fstat_time",
	"ko_time",
]

def get_covmethod_sample_kwargs(covmethod, args):
	max_corr = args.get('max_corr', [0.99])[0]
	sample_kwargs = {}
	if covmethod in ['ver', 'ar1']:
		sample_kwargs['max_corr'] = max_corr
	if covmethod == 'ar1':
		sample_kwargs['a'] = args.get("a", [5])[0]
		sample_kwargs['b'] = args.get("b", [1])[0]
	if covmethod == 'blockequi':
		sample_kwargs['rho'] = args.get("rho", [0.5])[0]
		sample_kwargs['gamma'] = args.get("gamma", [0])[0]
	if covmethod == 'ver':
		sample_kwargs['delta'] = args.get("delta", [0.2])[0]
	return sample_kwargs, max_corr

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
	try:
		sys.stdout.flush()
	except OSError:
		# Ignore stale file handle errors in cluster environments
		pass

	# 1. Create data-generating process for X
	output = []
	np.random.seed(seed)
	dgprocess = knockpy.dgp.DGP()
	sample_kwargs, max_corr = get_covmethod_sample_kwargs(covmethod, args)
	if covmethod not in ['orthogonal', 'ark']:
		dgprocess.sample_data(
			n=n,
			p=p,
			method=covmethod,
			**sample_kwargs,
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
		try:
			sys.stdout.flush()
		except OSError:
			# Ignore stale file handle errors in cluster environments
			pass

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
								corr_signals=args.get("corr_signals", [False])[0]
							)
							# Sample y
							if cond_mean == 'linear':
								fX, fXk = X, Xk
							elif cond_mean == 'cos':
								fX, fXk = np.cos(X), np.cos(Xk)
							elif cond_mean == 'sin':
								fX, fXk = np.sin(X), np.sin(Xk)
							elif cond_mean == 'cosh':
								fX, fXk = np.cosh(X), np.cosh(Xk)
							elif cond_mean == 'sinh':
								fX, fXk = np.sinh(X), np.sinh(Xk)
							elif cond_mean == 'quadratic':
								fX, fXk = X**2, Xk**2
							elif cond_mean == 'cubic':
								fX, fXk = X**3, Xk**3
							elif cond_mean == 'trunclin':
								fX = (X > 0).astype(float)
								fXk = (Xk > 0).astype(float)
							else:
								raise ValueError(f"unrecognized cond_mean={cond_mean}")
							mu = np.dot(fX, beta)
							y = gen_data.sample_y(mu=mu, y_dist=y_dist)

							# Assemble feature statistics
							fstats = []
							# oracle
							if not mx:
								fstats.append((oracle.OracleFXStatistic(beta=beta), 'oracle'))
							else:
								fstats.append(
									(knockpy.mlr.OracleMLR(beta=beta,), 'oracle')
								)
							# Linear statistics
							if args.get("compute_lcd", [True])[0]:
								fstats.append(('lcd', 'lcd'))
							if args.get('compute_lsm', [True])[0]:
								fstats.append(('lsm', 'lsm'))
							if args.get("compute_studentized", [True])[0]:
								fstats.append((StudentizedLassoStatistic(), 'lcd_studentized'))
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
								if fstat in ['mlr', 'bcd'] or (fstat == 'oracle' and mx):
									fstat_kwargs['n_iter'] = args.get("n_iter", [2000])[0]
									fstat_kwargs['chains'] = args.get("chains", [5])[0]

								# Quick kwarg to only run oracle statistic
								if args.get("oracle_only", [False])[0] and fstatname != 'oracle':
									continue

								# Run KF
								time0 = time.time()
								if fstatname != 'oracle' or not mx: 
									kf.forward(
										X=X, Xk=Xk, y=y, fstat_kwargs=fstat_kwargs, fdr=0.05,
									)
								else:
									# oracle statistics get to know what y is a linear response to
									kf.forward(
										X=fX, Xk=fXk, y=y, fstat_kwargs=fstat_kwargs, fdr=0.05,
									)
								fstat_time = time.time() - time0

								for q in args.get("q", [0.05, 0.10, 0.15, 0.20]):
									# Possibly compute MLR and AMLR statistics.
									# This is done this way to save time but this logic
									# does not affect the other statistics
									Ws = [kf.W]
									fnames = [fstatname]
									if fstatname == 'mlr':
										# hack to compute AMLR statistics at this FDR level
										kf.fstat.adjusted_mlr = True
										kf.fstat.fdr = q
										Ws.append(kf.fstat.compute_W())
										fnames.append("amlr")

									for W, fname in zip(Ws, fnames):
										T = kstats.data_dependent_threshhold(W=W, fdr=q)
										rej = (W >= T).astype("float32")
										power, fdp = utilities.calc_power_fdr(
											rej,
											beta,
										)

										# Add output
										output.append([
											n,
											p,
											covmethod,
											sparse,
											coeff_size,
											coeff_dist,
											seed,
											cond_mean,
											y_dist,
											mx,
											S_method,
											fname,
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

	# keywords for the data-generating process for X
	seed_start = args.get("seed_start", [0])[0]
	reps = args.get('reps', [1])[0] # number of replications
	num_processes = args.get('num_processes', [1])[0]
	ps = args.get('p', [300]) # dimensionality
	kappas = args.get('kappa', [2.1, 3, 4]) # ratio of n/p
	covmethods = args.get('covmethod', ['AR1']) # cov matrix
	job_id = args.get('job_id', [0])[0]

	# Save args, create output dir
	output_dir = utilities.create_output_directory(args, dir_type=DIR_TYPE)
	output_path = output_dir + f'job_id{job_id}_seedstart{seed_start}_results.csv'
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
						['mx', 'knockoff_type', 'n', 'p', 'covmethod', 'cond_mean',
						 'sparsity', 'coeff_size', 'q', 'fstat'] # 'lambd' for plug-in ests
					)[[
						'power', 'fdp', 'fstat_time', 'ko_time',
					]].agg(
						["mean", "std"]
					).reset_index()
					summary.to_csv(summary_path, index=False)
					pd.set_option('display.max_rows', 500)
					pd.set_option('display.max_columns', 20)
					print(summary)
					pd.reset_option('display.max_rows|display.max_columns|display.width')




if __name__ == '__main__':
	main(sys.argv)
