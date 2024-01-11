#!/bin/bash

set -e
set -x

ACTION_NUM=4
GROUP_NUM=2
PREF_DATA_NUM=20
PG_NUM_ITERS=1000
REG_COEF=0.01
STATE_DIM=1

# Initialize arrays to store rewards and weights
all_rewards=()
all_weights=()

# Generate weights from [0.1, 0.9] to [0.9, 0.1]
for weight in $(seq 0.1 0.1 0.9)
do
    WEIGHTS=[$weight,$(awk "BEGIN {print 1 - $weight}")]
    avg_reward=0

    for seed in 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030
    do
        reward=$(python -m experiments.run_group_linear_bandit \
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
            --logdir "log" | grep "reward error" | awk '{print $3}')

        avg_reward=$(awk "BEGIN {print $avg_reward + $reward}")

    done

    # Calculate the average reward
    avg_reward=$(awk "BEGIN {print $avg_reward / 10}")
    
    # Store results
    all_weights+=($weight)
    all_rewards+=($avg_reward)
done

# Print weights and average rewards
for idx in ${!all_weights[@]}; do
    echo "Weight: ${all_weights[$idx]}, Avg Reward: ${all_rewards[$idx]}"
done

# Plotting (example using matplotlib)
python -c "import matplotlib.pyplot as plt; plt.plot(${all_weights[@]}, ${all_rewards[@]}); plt.xlabel('Weights'); plt.ylabel('Average Reward'); plt.savefig('weights')"
