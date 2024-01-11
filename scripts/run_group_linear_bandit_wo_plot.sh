#!/bin/bash

set -e
set -x

ACTION_NUM=40
GROUP_NUM=2
PREF_DATA_NUM=20
PG_NUM_ITERS=1000
REG_COEF=0.01
STATE_DIM=1

# Create log directory with timestamp
LOG_DIR="log-weighted-dpo/$(date +'%Y%m%d%H%M%S')"
mkdir -p "$LOG_DIR"

# Generate weights from [0.1, 0.9] to [0.9, 0.1]
for weight in $(seq 0.0 0.1 1.0)
do
    WEIGHTS=[$weight,$(awk "BEGIN {print 1 - $weight}")]
    
    for seed in 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030
    do
        python -m experiments.run_group_linear_bandit \
        --mle_adaptive \
        --state_dim ${STATE_DIM} \
        --action_num ${ACTION_NUM} \
        --group_num ${GROUP_NUM} \
        --pref_data_num ${PREF_DATA_NUM} \
        --rl_data_ratio 0.5 \
        --pg_num_iters ${PG_NUM_ITERS} \
        --reg_coef ${REG_COEF} \
        --dpo_adaptive \
        --pg_adaptive \
        --seed ${seed} \
        --weights ${WEIGHTS} \
        --logdir ${LOG_DIR}
    done
done

#2022 2023 2024 2025 2026 2027 2028 2029 2030