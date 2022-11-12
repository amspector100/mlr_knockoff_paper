import os
import sys
import json
import numpy as np
import scipy as sp
from scipy import stats
import scipy.special

import time
from datetime import datetime
import pandas as pd
from pandas import DataFrame
import pandas_datareader.data as web

from context import mlr_src, knockpy
from mlr_src import parser
from knockpy.knockoff_filter import KnockoffFilter as KF
from knockpy import knockoff_stats as kstats

import warnings
import statsmodels.api as sm

def elapsed(time0):
	return np.around(time.time() - time0, 2)

def get_duplicate_columns(X):
	n, p = X.shape
	abscorr = np.abs(np.corrcoef(X.T))
	for j in range(p):
		for i in range(j+1):
			abscorr[i, j] = 0
	to_remove = np.where(abscorr > 1 - 1e-5)[0]
	return to_remove

def seqstep_plot(W, fdr=0.1):
	inds = np.argsort(-1*np.abs(W))
	sortW = W[inds]
	tau = knockpy.knockoff_stats.data_dependent_threshhold(sortW, fdr=fdr)
	tau = np.argmin(np.abs(sortW) >= tau)
	fig, ax = plt.subplots()
	x = np.arange(W.shape[0])
	ax.bar(x[sortW >= 0], sortW[sortW >= 0], color='blue')
	ax.bar(x[sortW < 0], sortW[sortW < 0], color='red')
	ax.axvline(tau, color='black', linestyle='dotted')
	plt.show()

def W_path_df(W):
	"""
	Given feature-statistics W creates a dataframe used for plotting.
	"""
	p = W.shape[0]
	inds = np.argsort(-1*np.abs(W), axis=0)
	sortW = np.take_along_axis(W, inds, axis=0)
	df = pd.DataFrame()
	df['W'] = sortW
	df['rank'] = np.arange(p)
	return df

def augment_X_y(X, y):
	n, p = X.shape
	if n < p:
		raise ValueError(f"Model is unidentifiable when n={n} < p={p}")
	if n > 2 * p:
		raise ValueError(f"n={n} > 2p={2*p}, no need to augment X and y")
	# Estimate sigma2
	ImH = np.eye(n) - X @ np.linalg.inv(X.T @ X) @ X.T
	hat_sigma2 = np.power(ImH @ y, 2).sum() / (n - p)
	# Sample new y, add zeros to X
	d = 2 * p - n
	ynew = np.concatenate(
		[y, np.sqrt(hat_sigma2) * np.random.randn(d)]
	)
	Xnew = np.concatenate(
		[X, np.zeros((d, p))],
		axis=0
	)
	return Xnew, ynew


COLUMNS = [
	'ndisc',
	'ndisc_true',
	'ndisc_false',
	'fstat',
	'S_method',
	'resistance',
	'drug_type'
]

def main(args):
	t0 = time.time()
	output = []
	W_output = []
	T_output = []
	# Common arguments with dependencies
	q = args.get("q", [0.05])[0]
	outfile = f"hiv_data/results/num_rej.csv"
	w_outfile = f"hiv_data/results/W_sorted.csv"
	t_outfile = f"hiv_data/results/thresholds.csv"
	# List of signal genes
	with open("hiv_data/signal_genes.json", 'r') as thefile:
		all_signal_genes = json.load(thefile)

	for drug_type in args.get("drug_types", ["PI", "NRTI", "NNRTI"]):
		# Signal genes
		sig_genes = set(all_signal_genes[drug_type][0])
		
		# Load raw data
		mutations = pd.read_csv(f"hiv_data/{drug_type}_mutations.csv")
		mutations = mutations.drop("Unnamed: 0", axis='columns')
		resistances = pd.read_csv(f"hiv_data/{drug_type}_resistances.csv")
		resistances = resistances.drop("Unnamed: 0", axis='columns')

		# Preprocess
		mutations = mutations[mutations.columns[mutations.sum(axis=0) >= 3]]

		# loop through y-values
		for col in args.get("resistances", resistances.columns):
			# Load y-data
			ycol = resistances[col]
			X = mutations.loc[ycol.notnull()].copy()
			y = np.log(ycol[ycol.notnull()].values)
			y = (y - y.mean()) / y.std()
			X = X[X.columns[X.sum(axis=0) >= 3]]

			# Remove duplicated columns
			duplicates = get_duplicate_columns(X.values)
			to_keep = [j for j in range(X.shape[1]) if j not in duplicates]
			Xcols = X.columns[to_keep]
			X = X[Xcols].values
			X = X.astype(float)
			X = (X - X.mean(axis=0)) / X.std(axis=0) # subtracting the mean makes some difference
			n, p = X.shape
			if n < 2*p:
				X, y = augment_X_y(X, y)

			# Run knockoffs
			for S_method in args.get("s_method", ["mvr", "sdp"]):
				idd = f"drug_type={drug_type}, resistance={col}, S_method={S_method}"
				print(f"Starting {idd} at {elapsed(t0)}.")
				np.random.seed(12345)
				ksampler = knockpy.knockoffs.FXSampler(
					X=X, method=S_method, tol=1e-3 if S_method=='sdp' else 1e-5
				)
				ksampler.sample_knockoffs()
				print(f"Finished sampling knockoffs for {idd} at {elapsed(t0)}.")
				kfilter = KF(fstat='mlr', ksampler=ksampler)
				rej_mlr = kfilter.forward(
					X=X, 
					y=y, 
					fdr=q, 
					fstat_kwargs={"n_iter":2000, "chains":5}#, "tau2_a0":2, "tau2_b0":0.01}
				)
				print(f"Finished fitting MLR statistic for {idd} at {elapsed(t0)}.")
				kfilter_lcd = KF(fstat='lcd', ksampler=ksampler)
				rej_lcd = kfilter_lcd.forward(X=X, y=y, fdr=q)
				print(f"Finished fitting LCD statistic for {idd} at {elapsed(t0)}.")
				kfilter_lsm = KF(fstat='lsm', ksampler=ksampler)
				rej_lsm = kfilter_lsm.forward(X=X, y=y, fdr=q)
				print(f"Finished fitting LSM statistic for {idd} at {elapsed(t0)}.")
				for fname, rej, kf in zip(
					['MLR', 'LSM', 'LCD'],
					[rej_mlr, rej_lsm, rej_lcd],
					[kfilter, kfilter_lsm, kfilter_lcd],
				):
					# Calculate true / false discoveries
					disc = Xcols[rej.astype(bool)].str.split(".")
					disc = set(disc.map(lambda x: int(x[0][1:])).tolist())
					ndisc = len(disc)
					ndisc_true = len([x for x in disc if x in sig_genes])
					ndisc_false = ndisc - ndisc_true
					output.append([ndisc, ndisc_true, ndisc_false, fname, S_method, col, drug_type])

					# Record W-statistics
					norm = np.abs(kf.W).max()
					Wdf = W_path_df(kf.W / norm)
					Wdf['fstat'] = fname
					Wdf['knockoff_type'] = S_method
					Wdf['drug_type'] = drug_type
					Wdf['drug'] = col
					W_output.append(Wdf)

					# Threshold
					Tind = np.argmax(np.abs(Wdf['W'].values) < kf.threshold / norm)
					T_output.append(
						[fname, S_method, drug_type, col, Tind]
					)

				# Save W-statistics and thresholds
				W_out_df = pd.concat(W_output, axis='index')
				T_out_df = pd.DataFrame(
					T_output, columns=['fstat', 'knockoff_type', 'drug_type', 'drug', 'rank']
				)
				W_out_df.to_csv(w_outfile, index=False)
				T_out_df.to_csv(t_outfile, index=False)
				print(T_out_df)
				print(W_out_df)

				# Print cumsums so far
				out_df = pd.DataFrame(output, columns=COLUMNS)
				out_df.to_csv(outfile, index=False)
				print(out_df)
				print(out_df.groupby(['S_method', 'fstat'])[['ndisc', 'ndisc_true', 'ndisc_false']].sum())


				

		# sns.heatmap(np.corrcoef(mutations.T), cmap='RdBu', vmin=-1, vmax=1)
		# plt.title(f"Heatmap of correlations for drug_type={drug_type}")
		# plt.show()

if __name__ == '__main__':
	args = parser.parse_args(sys.argv)
	main(args)