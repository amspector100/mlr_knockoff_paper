#!/bin/bash
#SBATCH --job-name=main
#SBATCH --output=slurm_logs/main_%A_%a.out # %A=job ID, %a=task ID
#SBATCH --error=slurm_logs/main_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --partition=candes,stat,hns,normal
#SBATCH --cpus-per-task=1
#SBATCH --array=1-512:1

REPS=1 # make sure this matches the increment in the SBATCH --array line
NUM_PROCESSES=1

job_id=${SLURM_ARRAY_JOB_ID} 
seed_start=${SLURM_ARRAY_TASK_ID}

# Load any modules needed
module load gcc/14.2.0
module load openblas/0.3.28
source /home/users/aspector/mlr/.venv/bin/activate


LINEAR_FX_ARGS="
        --p 500
        --covmethod [ar1,ver]
        --coeff_size 0.5
        --mx [False]
        --sparsity 0.1
        --kappa [2.05,2.5,3,3.5,4.0,4.5,5.0]
	--num_processes $NUM_PROCESSES
        --reps $REPS
        --job_id ${job_id}
        --seed_start ${seed_start}
"
LINEAR_MX_ARGS="
        --p 500
        --covmethod [ar1,ver]
        --coeff_size [0.5,1]
        --mx [True]
        --sparsity 0.1
        --kappa [0.25,0.5,0.75,1,1.25,1.5]
        --num_processes $NUM_PROCESSES
        --reps $REPS
        --job_id ${job_id}
        --seed_start ${seed_start}
        --oracle_only False
"
SPARSE_ARGS="
        --p 750
        --kappa 3
        --covmethod [ar1,ver]
        --coeff_size 0.3
        --mx [False]
        --coeff_dist [expo,uniform]
        --sparsity [0.05,0.1,0.2,0.3,0.4]
        --num_processes $NUM_PROCESSES
        --reps $REPS
        --job_id ${job_id}
        --seed_start ${seed_start}
"

NONLIN_ARGS="
        --p 200
        --covmethod [ar1]
        --mx [True]
        --sparsity [0.3]
        --coeff_size 1
        --kappa [3,5,10,15,20]
        --cond_mean [sin,cos,quadratic,cubic]
        --compute_lsm False
        --compute_mlr_spline True       
        --compute_randomforest True
        --compute_deeppink True
        --num_processes $NUM_PROCESSES
        --reps $REPS
        --job_id ${job_id}
        --seed_start ${seed_start}
        --n_iter 1000
        --chains 3
        --oracle_only False
"
VP_ARGS="
        --p [100,250,500,750,1000,1500,2000]
        --covmethod [ar1]
        --kappa [2.5]
        --coeff_size 0.5
        --sparsity [0.1]
        --mx [True,False]
        --n_iter 1000
        --chains 5
        --num_processes $NUM_PROCESSES
        --reps $REPS
        --job_id ${job_id}
        --seed_start ${seed_start}
"
LOGISTIC_ARGS="
        --y_dist logistic
        --p 500
        --covmethod [ver]
        --kappa [3,5,7,9]
        --mx [True]
        --coeff_size 1.0
        --n_iter 2000
        --chains 3
        --compute_lcd True
        --compute_lsm False
        --compute_mlr True
        --num_processes $NUM_PROCESSES
        --reps $REPS
        --job_id ${job_id}
        --seed_start ${seed_start}
"

AMLR_ARGS="
        --p 500
        --covmethod [ar1]
        --coeff_size [1]
        --mx [True]
        --sparsity 0.2
        --kappa [0.25,0.5,0.75,1,1.25,1.5]
        --compute_lcd False
        --compute_lsm False
        --n_iter 1000
        --reps $REPS
        --num_processes $NUM_PROCESSES
        --s_method mvr
        --job_id ${job_id}
        --seed_start ${seed_start}
"

AMLR_ARGS_FX="
        --p 500
        --covmethod ar1
        --coeff_size 0.5
        --mx False
        --sparsity 0.2
        --kappa [2.05,2.5,3,3.5,4.0,4.5,5.0]
        --compute_lcd False
        --compute_lsm False
        --n_iter 1000
        --reps $REPS
        --num_processes $NUM_PROCESSES
        --s_method mvr
        --job_id ${job_id}
        --seed_start ${seed_start}
"

#python3.9 sims_main.py $AMLR_ARGS_FX
python3.9 sims_main.py $LINEAR_FX_ARGS
python3.9 sims_main.py $LINEAR_MX_ARGS
#python3.9 sims_main.py $SPARSE_ARGS
#python3.9 sims_main.py $NONLIN_ARGS
#python3.9 sims_main.py $VP_ARGS
#python3.9 sims_main.py $LOGISTIC_ARGS
