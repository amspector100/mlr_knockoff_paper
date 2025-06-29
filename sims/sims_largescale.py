## main imports
import os
import sys
import time
import functools
import warnings
import numpy as np
from context import knockpy, mlr_src
from mlr_src import gen_data, oracle, parser, utilities
from mlr_src.utilities import elapsed
from mlr_src.studentized import StudentizedLassoStatistic
import knockpy
from knockpy import mlr
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from memory_profiler import profile

## Clean environment before importing R
# Remove problematic bash functions from python environment
bash_funcs_to_remove = [
	'BASH_FUNC__cache_cmd', 'BASH_FUNC_sacct', 'BASH_FUNC_squeue', 
	'BASH_FUNC_sinfo', 'BASH_FUNC_sstat', 'BASH_FUNC_sh_jobs',
	'BASH_FUNC_sh_quota', 'BASH_FUNC_sh_usage', 'BASH_FUNC_sh_jobwait',
	'BASH_FUNC_sh_part', 'BASH_FUNC_sh_next_downtime', 'BASH_FUNC_sudo',
	'BASH_FUNC_sleep', 'BASH_FUNC_bc', 'BASH_FUNC_sh_status', 
	'BASH_FUNC_ml', 'BASH_FUNC_module'
]

for func in bash_funcs_to_remove:
	if func in os.environ:
		del os.environ[func]

# # Suppress rpy2 warnings
# warnings.filterwarnings('ignore', category=UserWarning, module='rpy2')

## imports for R
import rpy2
from rpy2.robjects.numpy2ri import numpy2rpy
from rpy2.robjects.packages import importr

MAX_ITER = 100
# max n * p for lasso
MAX_NP_LASSO = 500000000
MAX_NP_LASSO_CV = 100000000
MAX_NP = int(500000 * 50000)

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

def suppress_warnings(func):
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			return func(*args, **kwargs)
	return wrapper

def gen_data(n, p):
	X = np.random.uniform(size=(n, p))
	beta = np.random.randn(p) / np.sqrt(p)
	Xk = np.random.uniform(size=(n, p))
	y = np.random.uniform(size=n) + X @ beta
	return dict(
		X=X, Xk=Xk, y=y
	)

@suppress_warnings
def lasso_cv_time_fit(X, Xk, y, n_folds=3):
	if MAX_NP_LASSO_CV < X.shape[0] * X.shape[1]:
		return np.nan
	features = np.concatenate([X, Xk], axis=1)
	t0 = time.time()
	lasso = LassoCV(cv=n_folds, copy_X=False, max_iter=MAX_ITER, n_alphas=10, n_jobs=-1)
	lasso.fit(features, y)
	del features
	return time.time() - t0

@suppress_warnings
def lasso_time_fit(X, Xk, y):
	if MAX_NP_LASSO < X.shape[0] * X.shape[1]:
		return np.nan
	t0 = time.time()
	lasso = Lasso(copy_X=False, max_iter=MAX_ITER)
	lasso.fit(np.concatenate([X, Xk], axis=1), y)
	return time.time() - t0

@suppress_warnings
def run_susie(
	X,
	y,
	L,
	q,
	**kwargs
):
	# import
	susieR = importr('susieR')
	t0 = time.time()
	# Run susie
	susieR.susie(
		X=numpy2rpy(X), y=numpy2rpy(y), L=L, coverage=1-q, **kwargs
	)
	return time.time() - t0

def compute_mlr_stats(data, n_iter, chains):
	t0 = time.time()
	mlr = knockpy.mlr.MLR_Spikeslab(
		n_iter=n_iter,
		chains=chains,
	)
	mlr.fit(groups=None, **data)
	return time.time() - t0

#@profile
def main():
	t0 = time.time()
	args = parser.parse_args(sys.argv)
	# parse key arguments
	seed_start = args.pop("seed_start", [1])[0]
	reps = args.pop("reps", [1])[0]
	job_id = args.pop("job_id", [0])[0]
	seeds = np.arange(seed_start, seed_start + reps)

	# Save args, create output dir
	output_dir = utilities.create_output_directory(args, dir_type=DIR_TYPE)
	output_path = output_dir + f'jobid{job_id}_seedstart{seed_start}_results.csv'
	
	# loop through and time
	output = []
	for seed in seeds:
		for p in args.get('p', [10000]):
			for n in args.get('n', [4000]):
				if n * p > MAX_NP:
					print(f"Skipping n={n}, p={p}, too expensive.")
				np.random.seed(seed)
				tdata = time.time()
				data = gen_data(n=n, p=p)
				preamble = f"For p={p}, n={n}, seed={seed}"
				print(f"{preamble}: sampling data took {elapsed(tdata)}, total_time={elapsed(t0)}.")
				lasso_time = lasso_time_fit(**data)
				print(f"{preamble}: lasso took {lasso_time}, total_time={elapsed(t0)}.")
				lasso_cv_time = lasso_cv_time_fit(**data)
				print(f"{preamble}: lasso cv took {lasso_cv_time}, total_time={elapsed(t0)}.")
				susie_time = run_susie(X=data['X'], y=data['y'], L=10, q=0.05)
				print(f"{preamble}: susie took {susie_time}, total_time={elapsed(t0)}.")
				mlr_time = compute_mlr_stats(data, n_iter=MAX_ITER, chains=1)
				print(f"{preamble}: mlr took {mlr_time}, total_time={elapsed(t0)}.")
				output.append(
					{
						"p":p,
						"n":n,
						"seed":int(seed),
						"mlr_time": mlr_time, 
						"lasso_time": lasso_time, 
						"lasso_cv_time": lasso_cv_time, 
						"susie_time": susie_time,
					}
				)
				out_df = pd.DataFrame(output)
				out_df.to_csv(output_path, index=False)
				# create summary
				summary = out_df.groupby(['p', 'n'])[['mlr_time', 'susie_time', 'lasso_cv_time', 'lasso_time']].mean()
				print(summary.reset_index())


if __name__ == "__main__":
	main()