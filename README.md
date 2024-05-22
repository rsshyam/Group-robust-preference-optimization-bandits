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

### Generating Paper Plots

The main plotting scripts are (1) ```plotting/wandb_data_collect_script_final_simplified.py``` and (2) ```plotting/wandb_data_collect_script_final_simplified_paperplots.py```.

Script (1) generates plots as individual ```plt.figure``` objects. Script (2) generates a ```plt.subplots``` object with multiple aligned subplots. This is how we present in the paper the synthetic-experiment plots of worst-case GR-DPO validation loss, worst-case GR-IPO validation loss, and GR-DPO/GR-IPO converged reward error. In script (2), the metrics being plotted together in the subplot must be manually-specified.

When running experiments (_Training and Evaluation_), the WandB Group name should match the group names of the runs that are downloaded by the plot scripts (i.e. the WandB groups defined in the ```SETTINGS``` var -- Line 20, scripts (1) and (2).).

Also, we include other plotting scripts that we used for ablation results.

