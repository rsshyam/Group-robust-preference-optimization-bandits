#!/bin/bash

set -e
set -x

ACTION_NUM=4
GROUP_NUM=2
PREF_DATA_NUM=20
PG_NUM_ITERS=1000
STATE_DIM=1
DPO_NUM_ITERS=50000

# Default values
DPO_TYPE='rdpo'
BATCH_SIZE=5
EXP_STEP_SIZE=0.01
FEATURE_TYPE='flipped'
WEIGHTED_BATCHES='true'
RDPO_ADJ='0'
EVAL_METRIC='expectation'
IMPORTANCE_SAMPLING='False'
IMPORTANCE_SAMPLING_WEIGHTS=None

# Parse command-line options
TEMP=$(getopt -o t:s:b:e:f: --long dpo_type:,batch_size:,exp_step_size:,feature_type:,weighted_batches:,rdpo_adj:,eval_metric:,importance_sampling:,importance_sampling_weights: -n 'your_script.sh' -- "$@")
if [ $? -ne 0 ]; then
    echo "Terminating..." >&2
    exit 1
fi
eval set -- "$TEMP"

# Process options
while true; do
  case "$1" in
    -t|--dpo_type) DPO_TYPE="$2"; shift 2 ;;
    -s|--batch_size) BATCH_SIZE="$2"; shift 2 ;;
    -b|--exp_step_size) EXP_STEP_SIZE="$2"; shift 2 ;;
    -e|--feature_type) FEATURE_TYPE="$2"; shift 2 ;;
    -f|--weighted_batches) WEIGHTED_BATCHES="$2"; shift 2 ;;
    --rdpo_adj) RDPO_ADJ="$2"; shift 2;;
    --eval_metric) EVAL_METRIC="$2"; shift 2;;
    --importance_sampling) IMPORTANCE_SAMPLING="$2"; shift 2;;
    --importance_sampling_weights) IMPORTANCE_SAMPLING_WEIGHTS="$2"; shift 2;;
    --) shift; break ;;
    *) echo "Internal error!" >&2; exit 1 ;;
  esac
done

# Create log directory with timestamp
LOG_DIR="log-weighted-dpo_sep/rdpo/$(date +'%Y_%m_%d_%H_%M_%S')_$DPO_NUM_ITERS"
mkdir -p "$LOG_DIR"



# Step_size values
STEP_SIZES=("0.1" "1.0" "0.01")
REG_COEFS=("5.0" "0.5" "0.1" "0.01" "0.001")

for STEP_SIZE in "${STEP_SIZES[@]}"; do
  for REG_COEF in "${REG_COEFS[@]}"; do

    # Generate weights from [0.1, 0.9] to [0.9, 0.1]
    for weight in $(seq 0.1 0.1 0.2); do
      WEIGHTS=[$weight,$(awk "BEGIN {print 1 - $weight}")]

      for seed in 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040; do
        python -m experiments.run_group_linear_bandit_sep_theta_combined \
          --mle_adaptive \
          --state_dim ${STATE_DIM} \
          --action_num ${ACTION_NUM} \
          --group_num ${GROUP_NUM} \
          --pref_data_num ${PREF_DATA_NUM} \
          --rl_data_ratio 0.5 \
          --pg_num_iters ${PG_NUM_ITERS} \
          --dpo_num_iters ${DPO_NUM_ITERS} \
          --wandb_use \
          --reg_coef ${REG_COEF} \
          --pg_adaptive \
          --seed ${seed} \
          --weights ${WEIGHTS} \
          --logdir ${LOG_DIR} \
          --dpo_type ${DPO_TYPE} \
          --dpo_step_size ${STEP_SIZE} \
          --rdpo_exp_step_size ${EXP_STEP_SIZE} \
          --rdpo_batch_size ${BATCH_SIZE} \
          --feature_type ${FEATURE_TYPE} \
          --rdpo_weighted_batches ${WEIGHTED_BATCHES} \
          --rdpo_adj ${RDPO_ADJ} \
          --eval_metric ${EVAL_METRIC} \
          --importance_sampling ${IMPORTANCE_SAMPLING} \
          --importance_sampling_weights ${IMPORTANCE_SAMPLING_WEIGHTS}
      done
    done
  done
done
