#!/bin/bash
#SBATCH --job-name=groupknock
#SBATCH --output=slurm_logs/groupknock_%A_%a.out # %A=job ID, %a=task ID
#SBATCH --error=slurm_logs/groupknock_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --partition=candes,stat,hns,normal
#SBATCH --cpus-per-task=1
#SBATCH --array=1-128:1

REPS=1 # make sure this matches the increment in the SBATCH --array line

job_id=${SLURM_ARRAY_JOB_ID} 
seed_start=${SLURM_ARRAY_TASK_ID}

# Load any modules needed
module load gcc/14.2.0
module load openblas/0.3.28
source /home/users/aspector/mlr/.venv/bin/activate

GROUPKNOCK_ARGS="
        --p 500
        --covmethod [ar1]
        --coeff_size 0.5
        --sparsity 0.1
        --correlation_cutoff [1,0.9,0.8,0.7,0.6]
        --n [250,375,500,625]
        --num_processes 1
        --reps ${REPS}
        --job_id ${job_id}
        --seed_start ${seed_start}
"

python sims_revisions.py $GROUPKNOCK_ARGS
 
echo "Task ${seed_start} finished with exit code $?"
