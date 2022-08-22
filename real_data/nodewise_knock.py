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

def elapsed(t0):
	return np.around(time.time() - t0, 2)

def main(args):
	t0 = time.time()

	# Load data
	data = pd.read_csv("nodewise_knock/ggm_data.csv")
	data = data.drop("Unnamed: 0", axis='columns')
	data = np.log(data + 1)
	# Select k with top variance
	k = args.get("k", [50])[0]
	stds = data.std(axis=0).sort_values(ascending=False)
	topkvar = stds.index[0:k]
	Xdata = data[topkvar]
	Xdata = (Xdata - Xdata.mean(axis=0)) / Xdata.std(axis=0)
	# Loop through S methods
	for S_method in args.get("s_method", ['mvr', 'sdp']):
		print(f"Starting computation for {S_method} knockoffs at {elapsed(t0)}.")
		# Loop through and compute W statistics
		W_lcd = []
		W_lsm = []
		W_mlr = []
		for j, col in enumerate(topkvar):
			# Extract X and y
			print(f"At j={j}, col={col} at {elapsed(t0)}.")
			negj = [i for i in range(k) if i != j]
			X = Xdata[topkvar[negj]].values
			y = Xdata[col].values
			# Create FX knockoffs
			ksampler = knockpy.knockoffs.FXSampler(X=X, method=S_method)
			S = ksampler.S
			# Loop through different feature stats
			mlrstat = knockpy.mlr.MLR_FX_Spikeslab(
			    tau2_a0=[2,2,2,2,2], tau2_b0=[0.01,0.1,0.5,1,10],
			    num_mixture=5,
			)
			for fstat, all_W in zip(
				['lcd', 'lsm', mlrstat], [W_lcd, W_lsm, W_mlr]
			):
				kf = KF(ksampler=ksampler, fstat=fstat)
				kf.forward(X=X, y=y)
				W = np.zeros(k)
				W[negj] = kf.W
				all_W.append(W)
		# Return
		for fstatname, all_W in zip(
			['lcd', 'lsm', 'mlr'], [W_lcd, W_lsm, W_mlr]
		):
			W_df = pd.DataFrame(all_W, index=topkvar, columns=topkvar)
			W_df.to_csv(f"nodewise_knock/results/W_{fstatname}_{S_method}_top{k}.csv")

if __name__ == '__main__':
	args = parser.parse_args(sys.argv)
	main(args)
