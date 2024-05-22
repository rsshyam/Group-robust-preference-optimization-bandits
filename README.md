# Group Robust Preference Optimization in Reward-free RLHF

This repository is the official implementation of the paper _Group Robust Preference Optimization in Reward-free RLHF_.


The main scripts relevant for group robust DPO and IPO are `scripts/run_glp_mult.sh` which calls upon `experiments/run_group_linear_bandit_sep_theta_combined_det_ratio`.

They use the Group Linear Bandit environment in `envs/group_linear_bandit.py`. The algorithms used are group DPO/IPO written together in `algos/linear_bandit/group_dpo.py` and group robust DPO/IPO in `algos/linear_bandit/group_robust_dpo.py`.

The `algos/vectorized_dpo.py` improves the DPO implementation in terms of vectorization and linear gradients avoiding numerical instability.

##  Requirements

The Python environment can be set up using Anaconda with the provided `environment.yml` file.

```
conda env create -f environment.yml
conda activate bandit
```

## Training and Evaluation


```
bash scripts/run_linear_bandit.sh
```

## Results

Explain how to obtain results from the paper



WandB Group name in running experiment should correspond to group names when running the plot script