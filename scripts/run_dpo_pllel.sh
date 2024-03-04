#!/bin/bash

set -e
set -x

ACTION_NUM=16
GROUP_NUM=2
PREF_DATA_NUM=40
PG_NUM_ITERS=1000
REG_COEF=0.01
STATE_DIM=1
DPO_NUM_ITERS=10000

# Define an array of step sizes in powers of 10
STEP_SIZES=(0.000001)

# Function to run the Python script for a given weight and seed
run_script() {
    local WEIGHT=$1
    local SEED=$2

    python -m experiments.run_group_linear_bandit_sep_theta \
        --mle_adaptive \
        --state_dim $STATE_DIM \
        --action_num $ACTION_NUM \
        --group_num $GROUP_NUM \
        --pref_data_num $PREF_DATA_NUM \
        --rl_data_ratio 0.5 \
        --pg_num_iters $PG_NUM_ITERS \
        --dpo_num_iters $DPO_NUM_ITERS \
        --reg_coef $REG_COEF \
        --pg_adaptive \
        --seed $SEED \
        --weights $WEIGHT \
        --logdir $LOG_DIR \
        --dpo_step_size $STEP_SIZE
}

# Iterate over step sizes
for STEP_SIZE in "${STEP_SIZES[@]}"
do
    # Create log directory with timestamp
    LOG_DIR=log-weighted-dpo_sep/dpo/$(date +'%Y_%m_%d_%H_%M_%S')_${DPO_NUM_ITERS}_${STEP_SIZE}
    mkdir -p "$LOG_DIR"

    # Iterate over weights in parallel
    for weight in $(seq 0.1 0.1 0.2)
    do
        # Iterate over seeds
        for seed in {2021..2022}
        do
            # Run the Python script in the background
            run_script $weight $seed &
        done
    done

    # Wait for all background processes to finish before moving to the next step size
    wait
done

echo "Script execution completed."
