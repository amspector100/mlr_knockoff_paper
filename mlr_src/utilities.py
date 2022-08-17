"""
Utility functions for simulations.
"""

import os
import sys
import time
import datetime
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial

def elapsed(t0):
	return np.around(time.time() - t0, 2)

### Multiprocessing helper
def _one_arg_function(list_of_inputs, args, func, kwargs):
	"""
	Globally-defined helper function for pickling in multiprocessing.
	:param list of inputs: List of inputs to a function
	:param args: Names/args for those inputs
	:param func: A function
	:param kwargs: Other kwargs to pass to the function. 
	"""
	new_kwargs = {}
	for i, inp in enumerate(list_of_inputs):
		new_kwargs[args[i]] = inp
	return func(**new_kwargs, **kwargs)


def apply_pool(func, constant_inputs={}, num_processes=1, **kwargs):
	"""
	Spawns num_processes processes to apply func to many different arguments.
	This wraps the multiprocessing.pool object plus the functools partial function. 
	
	Parameters
	----------
	func : function
		An arbitrary function
	constant_inputs : dictionary
		A dictionary of arguments to func which do not change in each
		of the processes spawned, defaults to {}.
	num_processes : int
		The maximum number of processes spawned, defaults to 1.
	kwargs : dict
		Each key should correspond to an argument to func and should
		map to a list of different arguments.
	Returns
	-------
	outputs : list
		List of outputs for each input, in the order of the inputs.
	Examples
	--------
	If we are varying inputs 'a' and 'b', we might have
	``apply_pool(
		func=my_func, a=[1,3,5], b=[2,4,6]
	)``
	which would return ``[my_func(a=1, b=2), my_func(a=3,b=4), my_func(a=5,b=6)]``.
	"""

	# Construct input sequence
	args = sorted(kwargs.keys())
	num_inputs = len(kwargs[args[0]])
	for arg in args:
		if len(kwargs[arg]) != num_inputs:
			raise ValueError(f"Number of inputs differs for {args[0]} and {arg}")
	inputs = [[] for _ in range(num_inputs)]
	for arg in args:
		for j in range(num_inputs):
			inputs[j].append(kwargs[arg][j])

	# Construct partial function
	partial_func = partial(
		_one_arg_function, args=args, func=func, kwargs=constant_inputs,
	)

	# Don't use the pool object if num_processes=1
	num_processes = min(num_processes, len(inputs))
	if num_processes == 1:
		all_outputs = []
		for inp in inputs:
			all_outputs.append(partial_func(inp))
	else:
		with Pool(num_processes) as thepool:
			all_outputs = thepool.map(partial_func, inputs)

	return all_outputs

def create_output_directory(args, dir_type='misc', return_date=False):
	# Date
	today = str(datetime.date.today())
	hour = str(datetime.datetime.today().time())
	hour = hour.replace(':','-').split('.')[0]
	# Output directory
	file_dir = os.path.dirname(os.path.abspath(__file__))
	parent_dir = os.path.split(file_dir)[0]
	output_dir = f'{parent_dir}/data/{dir_type}/{today}/{hour}/'
	# Ensure directory exists
	print(f"Output directory is {output_dir}")
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	# Save description
	args_path = output_dir + "args.json"
	with open(args_path, 'w') as thefile:
		thefile.write(json.dumps(args))
	# Return 
	if return_date:
		return output_dir, today, hour
	return output_dir

def calc_power_fdr(rejections, beta):
	# Calculate which features are not null
	nnulls = beta != 0
	# number of disc/false discoveries
	n_disc = np.sum(rejections)
	n_false_disc = np.sum(rejections * (1 - nnulls))
	# Power and fdr
	power = (n_disc - n_false_disc) / max(1, nnulls.sum())
	fdr = n_false_disc / max(1, n_disc)
	return (power, fdr)


def get_duplicate_columns(X):
	""" Helper for working with real data """
	n, p = X.shape
	abscorr = np.abs(np.corrcoef(X.T))
	for j in range(p):
	    for i in range(j+1):
	        abscorr[i, j] = 0
	to_remove = np.where(abscorr > 1 - 1e-5)[0]
	return to_remove