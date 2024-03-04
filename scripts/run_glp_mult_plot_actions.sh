#!/bin/bash

set -e
set -x

ACTION_NUM=4
GROUP_NUM=2
STATE_DIM=1
FEATURE_TYPE='same'
EVAL_METRIC='expectation'


# Parse command-line options
TEMP=$(getopt -o t:s:b:e:f: --long action_num:,group_num:,state_dim:,feature_type:,eval_metric: -n 'your_script.sh' -- "$@")
if [ $? -ne 0 ]; then
    echo "Terminating..." >&2
    exit 1
fi
eval set -- "$TEMP"

# Process options
while true; do
  case "$1" in
    -t|--action_num) ACTION_NUM="$2"; shift 2 ;;
    -s|--group_num) GROUP_NUM="$2"; shift 2 ;;
    -b|--state_dim) STATE_DIM="$2"; shift 2 ;;
    -e|--feature_type) FEATURE_TYPE="$2"; shift 2 ;;
    _f|--eval_metric) EVAL_METRIC="$2"; shift 2;;
    --) shift; break ;;
    *) echo "Internal error!" >&2; exit 1 ;;
  esac
done

# Create log directory with timestamp
LOG_DIR="action_plots/$ACTION_NUM$GROUP_NUM$STATE_DIM$FEATURE_TYPE$EVAL_METRIC"
mkdir -p "$LOG_DIR"

# Generate weights from [0.1, 0.9] to [0.9, 0.1]
for weight in $(seq 0.1 0.1 0.1)
do
    WEIGHTS=[$weight,$(awk "BEGIN {print 1 - $weight}")]
    
    for seed in 2021 
    do
        python -m experiments.plot_opt_actions \
        --state_dim ${STATE_DIM} \
        --action_num ${ACTION_NUM} \
        --group_num ${GROUP_NUM} \
        --feature_type ${FEATURE_TYPE} \
        --eval_metric ${EVAL_METRIC} 
    done
done

# 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040
# 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040