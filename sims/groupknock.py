"""
Main script for running simulations.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from context import knockpy, mlr_src
from mlr_src import parser, utilities
from mlr_src.utilities import elapsed
from knockpy import knockoffs
from knockpy.knockoff_filter import KnockoffFilter as KF
from knockpy import knockoff_stats as kstats

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

def single_seed_sim(
    seed,
    correlation_cutoff,
    t0,
    **args,
):
    """
    Assumptions:
    - Only MX knockoffs
    """
    output = []
    n = args.get("n", 500)
    p = args.get("p", 200)
    covmethod = args.get("covmethod", "AR1")
    sparsity = args.get("sparsity", 0.5)

    # 0. sample data
    np.random.seed(seed)
    dgprocess = knockpy.dgp.DGP()
    dgprocess.sample_data(
        n=n,
        p=p,
        method=covmethod,
        sparsity=sparsity,
    )
    X = dgprocess.X
    Sigma = dgprocess.Sigma
    y = dgprocess.y
    beta = dgprocess.beta
    
    # 1. create groups
    if correlation_cutoff==1:
        groups = np.arange(1, p + 1)
    else:
        groups = knockpy.dgp.create_grouping(Sigma, cutoff=correlation_cutoff)
    groups = groups.astype(int)
    
    # 2. create knockoffs
    S_methods = args.get("s_method", ['mvr', 'sdp'])
    for S_method in S_methods:
        time0 = time.time()
        ksampler = knockoffs.GaussianSampler(
            X=X, Sigma=Sigma, method=S_method, groups=groups
        )
        ksampler.fetch_S()
        Xk = ksampler.sample_knockoffs()
        ko_time = time.time() - time0
        print(f"Time taken to create {S_method} knockoffs for seed {seed}: {ko_time}, total time: {elapsed(t0)}")
        sys.stdout.flush()
        
        # 3. run knockoff filter for various statistics
        fstats = []
        if args.get("run_mlr", True):
            fstats.append("mlr")
        if args.get("run_lcd", True):
            fstats.append("lcd")
        if args.get("run_lsm", True):
            fstats.append("lsm")        
        for fstat in fstats:
            time0 = time.time()
            kf = KF(
                ksampler=ksampler,
                fstat=fstat,
            )
            kf.forward(
                X=X.astype(float),
                Xk=Xk.astype(float),
                y=y.astype(float),
                groups=groups.astype(int),
                fdr=0.05,
            )
            fstat_time = time.time() - time0
            print(f"fstat time for {S_method}, {fstat} seed {seed}: {fstat_time}, total time: {elapsed(t0)}")


            for q in args.get("q", [0.05, 0.10, 0.15, 0.20]):
                T = kstats.data_dependent_threshhold(W=kf.W, fdr=q)
                rej = (kf.W >= T).astype("float32")
                power, fdp = utilities.calc_power_fdr(rej, beta, groups=groups)

                # Add output
                output.append({
                    "n": n,
                    "p": p,
                    "knockoff_type": S_method,
                    "fstat": fstat,
                    "q": q,
                    "power": power,
                    "fdp": fdp,
                    "fstat_time": fstat_time,
                    "ko_time": ko_time,
                    "covmethod": covmethod,
                    "sparsity": sparsity,
                })

    return output

def main(args):

    # Parse and extract arguments
    t0 = time.time()

    # keywords for the data-generating process for X
    seed_start = args.pop("seed_start", [0])[0]
    reps = args.pop('reps', [1])[0] # number of replications
    num_processes = args.pop('num_processes', [1])[0]
    args['seed'] = list(range(seed_start, seed_start + reps))
    args['correlation_cutoff'] = args.get('correlation_cutoff', [0.9])

    # Save args, create output dir
    output_dir = utilities.create_output_directory(args, dir_type=DIR_TYPE)
    output_path = output_dir + 'results.csv'

    # Run and save output
    args.pop("description")
    all_outputs = utilities.apply_pool_factorial(
        func=single_seed_sim,
        constant_inputs={
            "t0": t0,
        },
        num_processes=num_processes,
        **args
    )
    output_df = []
    for output in all_outputs:
        output_df.extend(output)

    # Turn into pandas dataframe
    output_df = pd.DataFrame(output_df)
    output_df.to_csv(output_path, index=False)
    summary = output_df.groupby(
        ['knockoff_type', 'n', 'p', 'q', 'fstat'] # 'lambd' for plug-in ests
    )[[
        'power', 'fdp', 'fstat_time', 'ko_time',
    ]].agg(
        ["mean", "std"]
    ).reset_index()
    pd.set_option('display.max_rows', 500)
    print(summary)
    pd.reset_option('display.max_rows|display.max_columns|display.width')

if __name__ == "__main__":
    main(args=parser.parse_args(sys.argv))