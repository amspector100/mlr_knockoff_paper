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
from concurrent import futures

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

now_time = datetime.now()
start_time = datetime(year=2013, month=2, day=8)
FUNDS = [
	'XLB', # materials
	'XLC', # communication services
	'XLE', # energy sector
	'XLF', # financials
	'XLI', # industrials
	'XLK', # technology sector
	'XLP', # consumer staples 
	'XLRE', # real estate
	'XLU', # utility sector
	'XLV', # healthcare 
	'XLY', # consumer discretionary
]
SECTORS = {
	"XLB":"Materials",
	"XLC":"Communication Services",
	"XLE":"Energy",
	"XLF":"Financials",
	"XLI":"Industrials",
	"XLK":"Information Technology",
	"XLP":"Consumer Staples",
	"XLRE":"Real Estate",
	"XLU":"Utilities",
	"XLV":"Health Care",
	"XLY":"Consumer Discretionary"
}
COLUMNS = [
	'ndisc',
	'ndisc_true',
	'ndisc_false',
	'q',
	'fstat',
	'S_method',
	'fund',
]

def download_stock(stock, prefix=''):
	""" 
	Adapted from: https://github.com/CNuge/kaggle-code
	try to query the iex for a stock, if failed note with print
	"""
	try:
		stock_df = web.DataReader(stock,'yahoo', start_time, now_time)
		stock_df['Name'] = stock
		output_name = f'fund_replication/{prefix}{stock}.csv'
		stock_df.to_csv(output_name)
	except:
		print(f'bad: {stock}')

def download_sp_stock(stock):
	download_stock(stock, prefix='sp_')

def get_sp_data():
	"""
	Adapted from: https://github.com/CNuge/kaggle-code
	"""
	sp_stocks = ['MMM', 'ABT', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADBE', 'AMD', 'AAP', 'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 
		   'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT',
		   'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'ANTM', 'AON', 'AOS', 'APA', 'AAPL', 'AMAT', 'APTV', 'ADM',
		   'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'BKR', 'BLL', 'BAC', 'BBWI', 'BAX', 'BDX', 'BRK.B',
		   'BBY', 'BIO', 'TECH', 'BIIB', 'BLK', 'BK', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'CHRW',
		   'CDNS', 'CZR', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CDAY',
		   'CERN', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX',
		   'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'COP', 'ED', 'STZ', 'COO', 'CPRT', 'GLW', 'CTVA', 'COST', 'CTRA',
		   'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DISCA',
		   'DISCK', 'DISH', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DRE', 'DD', 'DXC', 'EMN', 'ETN', 'EBAY', 'ECL',
		   'EIX', 'EW', 'EA', 'EMR', 'ENPH', 'ETR', 'EOG', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'EVRG', 'ES', 'RE', 'EXC', 'EXPE',
		   'EXPD', 'EXR', 'XOM', 'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FRC', 'FISV', 'FLT', 'FMC', 'F', 'FTNT', 'FTV',
		   'FBHS', 'FOXA', 'FOX', 'BEN', 'FCX', 'GPS', 'GRMN', 'IT', 'GNRC', 'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL', 'GPN', 'GS',
		   'GWW', 'HAL', 'HBI', 'HIG', 'HAS', 'HCA', 'PEAK', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM',
		   'HPQ', 'HUM', 'HBAN', 'HII', 'IEX', 'IDXX', 'INFO', 'ITW', 'ILMN', 'INCY', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU',
		   'ISRG', 'IVZ', 'IPGP', 'IQV', 'IRM', 'JKHY', 'J', 'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'KEYS', 'KMB',
		   'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LEG', 'LDOS', 'LEN', 'LLY', 'LNC', 'LIN', 'LYV', 'LKQ', 
		   'LMT', 'L', 'LOW', 'LUMN', 'LYB', 'MTB', 'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT',
		   'MRK', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI',
		   'NDAQ', 'NTAP', 'NFLX', 'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI', 'NSC', 'NTRS', 'NOC', 'NLOK', 'NCLH', 'NRG', 'NUE',
		   'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'OKE', 'ORCL', 'OGN', 'OTIS', 'PCAR', 'PKG', 'PH', 'PAYX', 'PAYC', 'PYPL', 
		   'PENN', 'PNR', 'PBCT', 'PEP', 'PKI', 'PFE', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 
		   'PRU', 'PTC', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 
		   'RMD', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS',
		   'SNA', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STE', 'SYK', 'SIVB', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TGT', 'TEL',
		   'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TWTR', 'TYL', 'TSN', 'UDR',
		   'ULTA', 'USB', 'UAA', 'UA', 'UNP', 'UAL', 'UNH', 'UPS', 'URI', 'UHS', 'VLO', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VFC', 'VIAC', 
		   'VTRS', 'V', 'VNO', 'VMC', 'WRB', 'WAB', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WU', 'WRK', 'WY', 
		   'WHR', 'WMB', 'WLTW', 'WYNN', 'XEL', 'XLNX', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS']
	
	# maximum thread number
	max_workers = 50
	workers = min(max_workers, len(sp_stocks))
	# with futures.ThreadPoolExecutor(workers) as executor:
	# 	res = executor.map(download_sp_stock, sp_stocks)

	# combine into one file
	all_data = []
	for stock in sp_stocks:
		try:
			data = pd.read_csv(f"fund_replication/sp_{stock}.csv")
			all_data.append(data)
		except FileNotFoundError:
			pass #print(f"Failed to find stock={stock}")

	all_data = pd.concat(all_data, axis='index')
	all_data.to_csv("fund_replication/all_sp_stocks.csv")

def main(args):
	# Set seed
	np.random.seed(12345)

	output = []
	nyears = args.get("nyears", [10])[0]
	minx_time = f'{2022 - nyears}-01-01'

	# Possibly pull the s&p data
	xfile = "fund_replication/all_sp_stocks.csv"
	if not os.path.exists(xfile):
		get_sp_data()

	# Load sector data
	sector_data = pd.read_csv("fund_replication/all_sp_sectors.csv").reset_index()	

	# Load x data
	yvar = args.get("yvar", ['open'])[0]
	xdata = pd.read_csv(xfile)
	xdata = xdata.loc[xdata['Date'] >= minx_time]
	xdata = xdata.pivot(index=['Date'], columns='Name', values='Open')
	flags = np.any(~xdata.notnull(), axis='index')
	xcols = sorted([c for c in xdata.columns if not flags[c]])
	xdata = xdata[xcols]
	## Modification 1
	#xreturns = xdata
	xreturns = pd.DataFrame(
		np.log(xdata.values[1:] / xdata.values[0:-1]),
		#xdata.values[1:] / xdata.values[0:-1],
		index=xdata.index[1:]
	)
	xreturns -= xreturns.mean(axis=0) ### new
	xdates = set(xdata.index.tolist())

	# Create knockoffs
	for S_method in args.get("s_method", ['mvr', 'sdp']):
		ksampler_full = knockpy.knockoffs.FXSampler(
			X=xreturns.values, method=S_method, tol=1e-5
		)
		for fund in args.get("funds", FUNDS):
			fund = str(fund).upper()
			print(f"At fund={fund}, S_method={S_method}")
			sector = SECTORS[fund]
			sector_stocks = sector_data.loc[sector_data['GICS Sector'] == sector, 'Symbol'].tolist()
			try:
				ydata = pd.read_csv(f"fund_replication/{fund}.csv")
			except FileNotFoundError:
				download_stock(fund, prefix='')
				ydata = pd.read_csv(f"fund_replication/{fund}.csv")

			# Ensure dates are lined up
			ydata = ydata.set_index("Date")
			ydates = set(ydata.index.tolist())
			dates = sorted(list(ydates.intersection(xdates)))
			# Cache the case where all data is available to avoid recomputing S
			if len(dates) == ydata.shape[0] and ydata.shape[0] == xdata.shape[0]:
				X = xreturns.values
				ksampler = ksampler_full
				yraw = ydata[yvar.capitalize()].values
			else:
				X = xreturns.loc[dates[1:]].values
				yraw = ydata.loc[dates, yvar.capitalize()].values
				if X.shape[0] < 2 * X.shape[1]:
					print(f"n < 2p for fund={fund}, augmenting X, y")
					continue
					# try:
					# 	X, yraw = augment_X_y(X, yraw)
					# except ValueError: 
					# 	print(f"For fund={fund}, X shape is {X.shape}, model is unidentifiable")
					# 	continue
				X -= X.mean(axis=0) ### new
				ksampler = knockpy.knockoffs.FXSampler(
					X=X, method=S_method, tol=1e-5
				)

			# Log and standardize y
			y = np.log(yraw[1:] / yraw[0:-1])
			#y = yraw[1:] / yraw[0:-1]
			y = y - y.mean()
			y = y / y.std()

			# Method 1: LSM
			kf_lsm = KF(ksampler=ksampler, fstat='lsm')
			kf_lsm.forward(X=X, y=y)

			# Method 2: LC
			kf_lcd = KF(ksampler=ksampler, fstat='lcd')
			kf_lcd.forward(X=X, y=y)

			# Method 3: fxstat
			mlrstat = knockpy.mlr.MLR_FX_Spikeslab(				
				tau2_a0=[2,2,2,2,2], 
				tau2_b0=[0.01,0.1,0.5,1,10],
				num_mixture=5
			)
			kf_mlr = KF(
				ksampler=ksampler, fstat=mlrstat,
			)
			kf_mlr.forward(X=X, y=y)
			for q in args.get("q", [0.05]):
				for kf, method in zip(
					[kf_lsm, kf_lcd, kf_mlr], ['lsm', 'lcd', 'mlr']
				):
					rej = kf.make_selections(W=kf.W, fdr=q)
					ndisc = rej.sum()
					disc = [c for i, c in enumerate(xcols) if rej[i]]
					ndisc_true = len(set(disc).intersection(set(sector_stocks)))
					ndisc_false = ndisc - ndisc_true
					output.append(
						[ndisc, ndisc_true, ndisc_false, q, method, S_method, fund]
					)
			out_df = pd.DataFrame(output, columns=COLUMNS)
			out_df.to_csv(f"fund_replication/results/kf_results_{nyears}yr.csv", index=False)
			print(out_df)
			print(out_df.groupby(['fstat', 'S_method', 'q'])[['ndisc', 'ndisc_true', 'ndisc_false']].sum())



if __name__ == '__main__':
	args = parser.parse_args(sys.argv)
	main(args)