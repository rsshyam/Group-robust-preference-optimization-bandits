#!/bin/bash

set -e
set -x

ACTION_NUM=4
PREF_DATA_NUM=20
PG_NUM_ITERS=1000
REG_COEF=0.5
STATE_DIM=1
flipped=false
dpo_num_iters=10000
dpo_step_size=0.01
#1.0

# Create log directory with timestamp
LOG_DIR="log-dpo/dpo/$(date +'%Y_%m_%d_%H_%M_%S')_$DPO_NUM_ITERS"
mkdir -p "$LOG_DIR"

for seed in 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030
do
    python -m experiments.run_linear_bandit \
    --mle_adaptive \
    --state_dim ${STATE_DIM} \
    --action_num ${ACTION_NUM} \
    --pref_data_num ${PREF_DATA_NUM} \
    --rl_data_ratio 0.5 \
    --pg_num_iters ${PG_NUM_ITERS} \
    --reg_coef ${REG_COEF} \
    --dpo_step_size ${dpo_step_size} \
    --dpo_num_iters ${dpo_num_iters} \
    --pg_adaptive \
    --seed ${seed} \
    --logdir ${LOG_DIR}
done