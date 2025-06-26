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
from mlr_src.studentized import StudentizedLassoStatistic
from knockpy import knockoffs
from knockpy.knockoff_filter import KnockoffFilter as KF
from knockpy import knockoff_stats as kstats
from sims_main import get_covmethod_sample_kwargs

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

def single_seed_sim(
    seed,
    t0,
    **args,
):
    """
    Assumptions:
    - Only MX knockoffs
    """
    output = []
    p = args.get("p", 500)
    if 'kappa' in args and 'n' not in args:
        n = int(np.ceil(args.get("kappa") * p))
    else:
        n = args.get("n", 1000)
    covmethod = args.get("covmethod", "AR1")
    sparsity = args.get("sparsity", 0.5)
    correlation_cutoff = args.get("correlation_cutoff", 0.9)
    coeff_size = args.get("coeff_size", 0.5)

    print(f"Running with args={args} at seed {seed}, total time: {elapsed(t0)}.")
    try:
        sys.stdout.flush()
    except OSError:
        # Ignore stale file handle errors in cluster environments
        pass

    # 0. sample data
    np.random.seed(seed)
    dgprocess = knockpy.dgp.DGP()
    sample_kwargs, _ = get_covmethod_sample_kwargs(covmethod, args)
    dgprocess.sample_data(
        n=n,
        p=p,
        method=covmethod,
        sparsity=sparsity,
        coeff_size=coeff_size,
        **sample_kwargs
    )
    X = dgprocess.X
    estimate_sigma = args.get("estimate_sigma", True)
    if estimate_sigma:
        print("Estimating Sigma...")
        time0 = time.time()
        if n > p + 2:
            Sigma = np.cov(X.T)
        else:
            Sigma, _ = knockpy.utilities.estimate_covariance(X)
        print(f"Time taken to estimate Sigma: {time.time() - time0}, total time: {elapsed(t0)}.")
    else:
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
    if isinstance(S_methods, str):
        S_methods = [S_methods]
    for S_method in S_methods:
        time0 = time.time()
        ksampler = knockoffs.GaussianSampler(
            X=X, Sigma=Sigma, method=S_method, groups=groups
        )
        ksampler.fetch_S()
        Xk = ksampler.sample_knockoffs()
        ko_time = time.time() - time0
        print(f"Time taken to create {S_method} knockoffs for seed {seed}: {ko_time}, total time: {elapsed(t0)}")
        try:
            sys.stdout.flush()
        except OSError:
            # Ignore stale file handle errors in cluster environments
            pass
        
        # 3. run knockoff filter for various statistics
        fstats = []
        fstat_kwargs = []
        if args.get("run_mlr", True):
            fstats.append(("mlr", "mlr"))
            fstat_kwargs.append(dict(n_iter=args.get("n_iter", [2000])[0], chains=args.get("chains", [5])[0]))
        if args.get("run_lcd", True):
            fstats.append(("lcd", "lcd"))
            fstat_kwargs.append({})
        if args.get("run_lsm", True):
            fstats.append(("lsm", "lsm"))  
            fstat_kwargs.append({})      
        if args.get("run_studentized", True):
            fstats.append(("lcd_studentized", StudentizedLassoStatistic()))
            fstat_kwargs.append({})
        for (fstat_name, fstat), fkwargs in zip(fstats, fstat_kwargs):
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
            print(f"fstat time for {S_method}, {fstat_name} seed {seed}: {fstat_time}, total time: {elapsed(t0)}")

            qs = args.get("q", [0.05, 0.10, 0.15, 0.20])
            if isinstance(qs, float):
                qs = [qs]
            for q in qs:
                T = kstats.data_dependent_threshhold(W=kf.W, fdr=q)
                rej = (kf.W >= T).astype("float32")
                power, fdp = utilities.calc_power_fdr(rej, beta, groups=groups)

                # Add output
                output.append({
                    "n": n,
                    "p": p,
                    "knockoff_type": S_method,
                    "fstat": fstat_name,
                    "q": q,
                    "power": power,
                    "fdp": fdp,
                    "fstat_time": fstat_time,
                    "ko_time": ko_time,
                    "covmethod": covmethod,
                    "sparsity": sparsity,
                    "correlation_cutoff": correlation_cutoff,
                    "ngroups": len(np.unique(groups)),
                    "estimate_sigma": estimate_sigma,
                    "coeff_size": coeff_size,
                    "seed": seed,
                })

    return output

def main(args):

    # Parse and extract arguments
    t0 = time.time()

    # keywords for the whole simulation
    num_processes = args.pop('num_processes', [1])[0]
    seed_start = args.pop("seed_start", [0])[0]
    reps = args.pop('reps', [1])[0] # number of replications
    args['seed'] = list(range(seed_start, seed_start + reps))
    args['correlation_cutoff'] = args.get('correlation_cutoff', [0.9])
    job_id = args.pop('job_id', [0])[0]

    # Save args, create output dir
    output_dir = utilities.create_output_directory(args, dir_type=DIR_TYPE)
    output_path = output_dir + f'jobid{job_id}_seedstart{seed_start}_results.csv'

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