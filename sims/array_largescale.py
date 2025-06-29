import subprocess
from datetime import datetime
import random
import argparse

MAX_REQ_MB = 1000000
SEED_START = 1
REPS = 1
REPS_PER_JOB = 1

def memory_requirement_mb(n, p):
    required_gb = 2 * n * p / (1e7)
    required_mb = max(int(1000 * required_gb), 4000)
    return required_mb

def num_cores(n, p):
    return 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=1)
    args = parser.parse_args()
    test = args.test
    
    minutes_since_year_start = int((datetime.now() - datetime(datetime.now().year, 1, 1)).total_seconds() / 60)
    job_id = 100 * minutes_since_year_start + random.randint(0, 100)
    print("Executing sbatch commands for job_id:", job_id)
    
    if test == 1:
        p_n_combos = [
            dict(p=50, n=20),
            dict(p=20, n=200),
        ]
        print(f"Running in test mode.")
    elif test == 2:
        p_n_combos = [
            dict(p=1000, n=5000),
        ]
        print("Running a real test.")
    else:
        ps = [2500, 5000, 10000, 20000, 40000]
        ns = [5000, 13000, 40000, 120000, 380000]
        p_n_combos = [dict(p=p, n=n) for p in ps for n in ns]
        print(f"Not running test mode.")

    for p_n in p_n_combos:
        p = p_n['p']
        n = p_n['n']
        # get memory requirement
        req_mb = memory_requirement_mb(n, p)
        if req_mb > MAX_REQ_MB:
            print(f"Memory requirement for n={n}, p={p} is {req_mb} MB, which is too large.")
            continue
        ncores = num_cores(n, p)
        # submit job
        sbatch_cmd = [
            "sbatch",
            "--job-name=mlr_largescale",
            f"--output=slurm_logs_ls/mlr_largescale_{job_id}_p{p}_n{n}_ncores{ncores}_mem{req_mb}.out",
            f"--error=slurm_logs_ls/mlr_largescale_{job_id}_p{p}_n{n}_ncores{ncores}_mem{req_mb}.err",
            "--partition=candes,candes,stat,hns,normal,bigmem",
            "--time=24:00:00",
            f"--mem={req_mb}M",
            f"--cpus-per-task={ncores}",
            "largescale.sh",
            f"--p {p}",
            f"--n {n}",
            f"--reps {REPS}",
            f"--job_id {job_id}",
            f"--seed_start {SEED_START}",
        ]
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
        print(f"Submitted job with command {sbatch_cmd}: output={result.stdout.strip()}")

if __name__ == "__main__":
    main()