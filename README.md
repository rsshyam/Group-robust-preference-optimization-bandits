# GRPO: Group Robust Preference Optimization

This repository implementats our experiments with synthetic preference data in the paper _Group Robust Preference Optimization in Reward-free RLHF_.


The main scripts relevant for group robust DPO and IPO are `scripts/run_glp_mult.sh` which calls upon `experiments/run_group_linear_bandit_sep_theta_combined_det_ratio`.

They use the Group Linear Bandit environment in `envs/group_linear_bandit.py`. The algorithms used are group DPO/IPO written together in `algos/linear_bandit/group_dpo.py` and group robust DPO/IPO in `algos/linear_bandit/group_robust_dpo.py`.

The `algos/vectorized_dpo.py` improves the DPO implementation in terms of vectorization and linear gradients avoiding numerical instability.

##  Requirements and Environment

The Python environment can be set up using Anaconda with the provided `environment.yml` file.

```
conda env create -f environment.yml
conda activate bandit
```

## Training and Evaluation

The main experiment file is ```experiments/run_group_linear_bandit_sep_theta_combined_uneven_grp_vectorised.py```. This file is called from ```scripts/run_glp_mult_uneven.sh```. Due to the large number of command-line arguments, we also provide scripts in ```scripts/scripts\ to\ reproduce```, which only alter significant arguments and call ```scripts/run_glp_mult_uneven.sh```.

The important arguments are:
- ```dpo_type``` : 'dpo' for non-robust baseline, and 'rdpo' for robust GRPO.
- ```ipo_grad_type``` : 'justdpo' for DPO loss (this is GR-DPO if ```dpo_type = rdpo``` or DPO/IS-DPO if ```dpo_type = rdpo```); 'linear' for IPO loss (GR-IPO if ```dpo_type = rdpo``` or IPO/IS-IPO if ```dpo_type = rdpo```)
- ```importance_sampling``` & ```importance_sampling_weights``` : If ```importance_sampling = true``` and ```importance_sampling_weights``` are given as a string-array, IS-DPO/IS-IPO. Elif ```importance_sampling = false```, DPO/IPO. Here, one must concurrently set ```dpo_type = dpo```.
- ```feature_type``` : 'swapped', 'flipped' or 'same' feature vector φ(x,y,g).
- ```weight``` : Percentage of training samples given to the first group (2 groups case); ```weight=0.2``` is imbalanced-data (20-80 imbalance), and ```weight=0.5``` is balanced-data (50-50).

Call ```scripts/scripts\ to\ reproduce``` scripts to run (1) Even-Imbalanced, (2) Uneven-Balanced, or (3) Uneven-Imbalanced configurations. The call is as follows

```
bash scripts/scripts\ to\ reproduce/even_group_imbalanced_data.sh
bash scripts/scripts\ to\ reproduce/uneven_group_balanced_data.sh
bash scripts/scripts\ to\ reproduce/uneven_group_imbalanced_data.sh
```

All synthetic experiments run for 20 seeds (2021-2040), in CPU compute (Intel Xeon E5-4620 v4 @ 2.10GHz & Xeon E5-2660 v3 @ 2.60GHz).

## Results

### Generating Plots from Paper

The main plotting scripts are (1) ```plotting/wandb_data_collect_script_final_simplified.py``` and (2) ```plotting/wandb_data_collect_script_final_simplified_paperplots.py```.

Script (1) generates plots as individual ```plt.figure``` objects. Script (2) generates a ```plt.subplots``` object with multiple aligned subplots. This is how we present in the paper the synthetic-experiment plots of worst-case GR-DPO validation loss, worst-case GR-IPO validation loss, and GR-DPO/GR-IPO converged reward error. In script (2), the metrics being plotted together in the subplot must be manually-specified.

When running experiments (_Training and Evaluation_), the WandB Group name should match the group names of the runs that are downloaded by the plot scripts (i.e. the WandB groups defined in the ```SETTINGS``` var -- Line 20, scripts (1) and (2).).

To run Script (2) for all configurations (even/uneven groups, balanced/imbalanced data), ```plotting/plotting.sh``` may be used. For each configuration, append ```_dpo``` in ```--setting``` for GR-DPO plots, ```_ipo``` for GR-IPO plots, and ```_all``` for GR-DPO & GR-IPO together.

```bibtex
@article{ramesh2024grpo,
  title={Group Robust Preference Optimization in Reward-free RLHF},
  author={Shyam Sundhar Ramesh, Iason Chaimalas, Viraj Mehta, Haitham Bou Ammar, Pier Giuseppe Sessa, Yifan Hu, Ilija Bogunovic},
  year={2024}
}
```