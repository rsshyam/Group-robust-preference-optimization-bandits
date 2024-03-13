#!/bin/bash

set -e
set -x

ACTION_NUM=8
GROUP_NUM=2
PREF_DATA_NUM=300 # change batch_size accordingly
BATCH_SIZE=300
PG_NUM_ITERS=1000
STATE_DIM=2


# Default values
DETERMINISTIC_RATIO_LIST='[1,1]'
VAL_DETERMINISTIC='True'
DPO_TYPE='dpo'
#'rdpo' # rdpo for RobustDPO & dpo for DPO (both VECTORISED)
STEP_SIZE=0.1
REG_COEF=0.1
#0.001
EXP_STEP_SIZE=0.05 # was 0.0001
FEATURE_TYPE='swapped'
#'flipped'
WEIGHTED_BATCHES='false'
RDPO_ADJ='0'
EVAL_METRIC='argmax'
IMPORTANCE_SAMPLING='False'
IMPORTANCE_SAMPLING_WEIGHTS=None
DETERMINISTIC_LIST='[False,False]'
IPO_GRAD_TYPE='justdpo'
PARAM_LIMIT=5
DPO_NUM_ITERS=20000
USE_CLOSED_FORM=False
L2_REG_RDPO=0.05
#REG_BY_GROUP_WEIGHTS=0.05
LAMBA=0

# Parse command-line options
TEMP=$(getopt -o t:s:b:e:f: --long dpo_type:,step_size:,reg_coef:,batch_size:,exp_step_size:,feature_type:,weighted_batches:,rdpo_adj:,eval_metric:,importance_sampling:,importance_sampling_weights:,ipo_grad_type:,param_limit:,dpo_num_iters:,use_closed_form:,val_deterministic:,lamba:,deterministic_ratio_list:,deterministic_list: -n 'your_script.sh' -- "$@")
if [ $? -ne 0 ]; then
    echo "Terminating..." >&2
    exit 1
fi
eval set -- "$TEMP"

# Process options
while true; do
  case "$1" in
    -t|--dpo_type) DPO_TYPE="$2"; shift 2 ;;
    -s|--step_size) STEP_SIZE="$2"; shift 2 ;;
    -b|--reg_coef) REG_COEF="$2"; shift 2 ;;
    -e|--batch_size) BATCH_SIZE="$2"; shift 2 ;;
    -f|--exp_step_size) EXP_STEP_SIZE="$2"; shift 2 ;;
    --feature_type) FEATURE_TYPE="$2"; shift 2 ;;
    --weighted_batches) WEIGHTED_BATCHES="$2"; shift 2 ;;
    --rdpo_adj) RDPO_ADJ="$2"; shift 2;;
    --eval_metric) EVAL_METRIC="$2"; shift 2;;
    --importance_sampling) IMPORTANCE_SAMPLING="$2"; shift 2;;
    --importance_sampling_weights) IMPORTANCE_SAMPLING_WEIGHTS="$2"; shift 2;;
    --ipo_grad_type) IPO_GRAD_TYPE="$2"; shift 2;;
    --param_limit) PARAM_LIMIT="$2"; shift 2;;
    --dpo_num_iters) DPO_NUM_ITERS="$2"; shift 2;;
    --use_closed_form) USE_CLOSED_FORM="$2"; shift 2;;
    --val_deterministic) VAL_DETERMINISTIC="$2"; shift 2;;
    --deterministic_ratio_list) DETERMINISTIC_RATIO_LIST="$2"; shift 2;;
    --deterministic_list) DETERMINISTIC_LIST="$2"; shift 2;;
    --l2_reg_rdpo) L2_REG_RDPO="$2"; shift 2;;
    --lamba) LAMBA="$2"; shift 2;;
    --) shift; break ;;
    *) echo "Internal error!" >&2; exit 1 ;;
  esac
done

# Create log directory with timestamp
LOG_DIR="log-weighted-dpo_sep_vectorised/rdpo/$(date +'%Y_%m_%d_%H_%M_%S')_$DPO_NUM_ITERS"
mkdir -p "$LOG_DIR"

# Generate weights from [0.1, 0.9] to [0.9, 0.1]
for weight in $(seq 0.1 0.1 0.1) # 0.1 0.1 0.9
do
    WEIGHTS=[$weight,$(awk "BEGIN {print 1 - $weight}")]
    
    for seed in 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031 2032 2033 2034 # 2021 #2035 2036 2037 2038 2039 2040
    do
        python -m experiments.run_group_linear_bandit_sep_theta_combined_det_ratio_vectorised \
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
        --importance_sampling_weights ${IMPORTANCE_SAMPLING_WEIGHTS} \
        --ipo_grad_type ${IPO_GRAD_TYPE} \
        --param_limit ${PARAM_LIMIT} \
        --use_closed_form ${USE_CLOSED_FORM} \
        --val_deterministic ${VAL_DETERMINISTIC} \
        --deterministic_ratio_list ${DETERMINISTIC_RATIO_LIST} \
        --l2_reg_rdpo ${L2_REG_RDPO} \
        --lamba ${LAMBA}
    done
done

# 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040
# 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040 
# 2021 2022 2023 2024 2025 2026 2027 2028 2029 