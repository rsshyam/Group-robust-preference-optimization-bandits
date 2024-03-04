# Policy Optimization in RLHF: The Impact of Out-of-preference Data


This repository improves upon the code for experiments in the [paper](https://arxiv.org/abs/2312.10584): Policy Optimization in RLHF: The Impact of Out-of-preference Data.

The main scripts relevant for group robust DPO and IPO are scripts/run_glp_mult.sh which calls upon experiments/run_group_linear_bandit_sep_theta_combined_det_ratio.

They use the Group Linear Bandit environment in envs/group_linear_bandit.py. The algorithms used are group DPO/IPO written together in algos/linear_bandit/group_dpo.py and group robust DPO/IPO in algos/linear_bandit/group_robust_dpo.py.

The algos/vectorized_dpo.py improves the DPO implementation in terms of vectorization and linear gradients avoiding numerical instability.



The experiments show that policy optimization with out-of-preference data is key to unlocking the reward model's generalization power.


<img src='./images/neural_bandit.png' width='600'>


##  How to use

### Prepare

The Python environment can be set up using Anaconda with the provided `environment.yml` file.

```
conda env create -f environment.yml
conda activate bandit
```

### Linear Bandit


```
bash scripts/run_linear_bandit.sh
```

### Neural Bandit


```
bash scripts/run_neural_bandit.sh
```

## Bibtex

If you find this code is helpful, please cite our paper in the following format.

```
@article{li2023policy,
  title     = {Policy Optimization in RLHF: The Impact of Out-of-preference Data},
  author    = {Li, Ziniu and Xu, Tian and Yu, Yang},
  journal   = {arXiv preprint arXiv:2312.10584},
  year      = {2023},
}
```

