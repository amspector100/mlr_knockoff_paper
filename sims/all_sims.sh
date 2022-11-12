REPS=10
NUM_PROCESSES=2

LINEAR_FX_ARGS="
        --p 500
        --covmethod [ar1,ver]
        --coeff_size 0.5
        --mx [False]
        --sparsity 0.1
        --kappa [2.05,2.5,3,3.5,4.0,4.5,5.0]
        --num_processes $NUM_PROCESSES
        --reps $REPS
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
        --oracle_only True
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
"

# cond means: [sin,cos,quadratic,cubic]

NONLIN_ARGS="
        --p 200
        --covmethod [ar1]
        --mx [True]
        --sparsity [0.3]
        --coeff_size 1
        --kappa [3,5,10,15,20]
        --cond_mean [cubic]
        --compute_lsm False
        --compute_mlr_spline True       
        --compute_randomforest True
        --compute_deeppink True
        --num_processes $NUM_PROCESSES
        --reps $REPS
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
        --reps 64
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
        --reps 94
"

#python3.9 main.py $SPARSE_ARGS
#python3.9 main.py $HEAVYTAIL_ARGS

python3.9 main.py $NONLIN_ARGS

#python3.9 main.py $LINEAR_FX_ARGS
#python3.9 main.py $LINEAR_MX_ARGS

#python3.9 main.py $VP_ARGS

#python3.9 main.py $LOGISTIC_ARGS

