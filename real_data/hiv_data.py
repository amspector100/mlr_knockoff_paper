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
	# Common arguments with dependencies
	q = args.get("q", [0.05])[0]
	outfile = f"hiv_data/results/num_rej.csv"
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
				kfilter2 = KF(fstat='lcd', ksampler=ksampler)
				rej_lcd = kfilter2.forward(X=X, y=y, fdr=q)
				print(f"Finished fitting LCD statistic for {idd} at {elapsed(t0)}.")
				kfilter3 = KF(fstat='lsm', ksampler=ksampler)
				rej_lsm = kfilter2.forward(X=X, y=y, fdr=q)
				print(f"Finished fitting LSM statistic for {idd} at {elapsed(t0)}.")
				for fname, rej in zip(
					['MLR', 'LSM', 'LCD'],
					[rej_mlr, rej_lsm, rej_lcd]
				):
					# Calculate true / false discoveries
					disc = Xcols[rej.astype(bool)].str.split(".")
					disc = set(disc.map(lambda x: int(x[0][1:])).tolist())
					ndisc = len(disc)
					ndisc_true = len([x for x in disc if x in sig_genes])
					ndisc_false = ndisc - ndisc_true
					output.append([ndisc, ndisc_true, ndisc_false, fname, S_method, col, drug_type])

				# Print cumsums so far
				out_df = pd.DataFrame(output, columns=COLUMNS)
				out_df.to_csv(outfile)
				print(out_df)
				print(out_df.groupby(['S_method', 'fstat'])[['ndisc', 'ndisc_true', 'ndisc_false']].sum())
			

		# sns.heatmap(np.corrcoef(mutations.T), cmap='RdBu', vmin=-1, vmax=1)
		# plt.title(f"Heatmap of correlations for drug_type={drug_type}")
		# plt.show()

if __name__ == '__main__':
	args = parser.parse_args(sys.argv)
	main(args)