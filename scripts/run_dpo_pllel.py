import subprocess
import os
from datetime import datetime

ACTION_NUM = 16
GROUP_NUM = 2
PREF_DATA_NUM = 40
PG_NUM_ITERS = 1000
REG_COEF = 0.01
STATE_DIM = 1
DPO_NUM_ITERS = 10000
STEP_SIZES = [0.000001]

for STEP_SIZE in STEP_SIZES:
    # Create log directory with timestamp
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    LOG_DIR = f"log-weighted-dpo_sep/dpo/{timestamp}_{DPO_NUM_ITERS}_{STEP_SIZE}"
    os.makedirs(LOG_DIR)

    # Iterate over weights
    for weight in [round(x * 0.1, 1) for x in range(1, 10)]:
        WEIGHTS = [weight, 1 - weight]

        # Iterate over seeds
        for seed in range(2021, 2041):
            command = [
                "python",
                "-m",
                "experiments.run_group_linear_bandit_sep_theta",
                "--mle_adaptive",
                "--state_dim", str(STATE_DIM),
                "--action_num", str(ACTION_NUM),
                "--group_num", str(GROUP_NUM),
                "--pref_data_num", str(PREF_DATA_NUM),
                "--rl_data_ratio", "0.5",
                "--pg_num_iters", str(PG_NUM_ITERS),
                "--dpo_num_iters", str(DPO_NUM_ITERS),
                "--reg_coef", str(REG_COEF),
                "--pg_adaptive",
                "--seed", str(seed),
                "--weights", str(WEIGHTS),
                "--logdir", LOG_DIR,
                "--dpo_step_size", str(STEP_SIZE),
            ]
            
            # Run the command in the background
            subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("Script execution completed.")
