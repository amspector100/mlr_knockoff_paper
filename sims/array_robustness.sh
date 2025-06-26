#!/bin/bash
#SBATCH --job-name=robustness
#SBATCH --output=slurm_logs/robustness_%A_%a.out # %A=job ID, %a=task ID
#SBATCH --error=slurm_logs/robustness_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --partition=candes,stat,hns,normal
#SBATCH --cpus-per-task=1
#SBATCH --array=1-56:1

REPS=1 # make sure this matches the increment in the SBATCH --array line

job_id=${SLURM_ARRAY_JOB_ID} 
seed_start=${SLURM_ARRAY_TASK_ID}

# Load any modules needed
module load gcc/14.2.0
module load openblas/0.3.28
source /home/users/aspector/mlr/.venv/bin/activate

ROBUSTNESS_ARGS="
        --p 500
        --covmethod [ar1,ver]
        --coeff_size [0.5]
        --mx [True]
        --sparsity 0.1
        --kappa [0.25,0.375,0.5,0.625,0.75,0.875,1,1.125,1.25,1.375,1.5]
        --correlation_cutoff 1
        --estimate_sigma true
        --num_processes 1
        --reps $REPS
        --job_id ${job_id}
        --seed_start ${seed_start}
"

python sims_revisions.py $ROBUSTNESS_ARGS
 
echo "Task ${seed_start} finished with exit code $?"
