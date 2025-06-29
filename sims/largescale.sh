#!/bin/bash

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n)
            n="$2"
            shift 2
            ;;
        --n=*)
            n="${1#*=}"
            shift
            ;;
        --p)
            p="$2"
            shift 2
            ;;
        --p=*)
            p="${1#*=}"
            shift
            ;;
        --job_id)
            job_id="$2"
            shift 2
            ;;
        --job_id=*)
            job_id="${1#*=}"
            shift
            ;;
        --seed_start)
            seed_start="$2"
            shift 2
            ;;
        --seed_start=*)
            seed_start="${1#*=}"
            shift
            ;;
        --reps)
            reps="$2"
            shift 2
            ;;
        --reps=*)
            reps="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done
echo "n=${n}, p=${p}, job_id=${job_id}, seed_start=${seed_start}, reps=${reps}"

# Load any modules needed
source /home/users/aspector/mlr/setup_env.sh

LARGESCALE_ARGS="
        --n ${n}
        --p ${p}
        --job_id ${job_id}
        --reps ${reps}
        --seed_start ${seed_start}
"

python sims_largescale.py $LARGESCALE_ARGS

echo "Task ${n} finished with exit code $?"

