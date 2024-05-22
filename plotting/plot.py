
import neatplot
neatplot.set_style()

import os
import yaml
import matplotlib.pyplot as plt
import numpy as np

def load_yml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def plot_grouped_bar_graph(weights, average_errors, num_groups):
    x = np.arange(len(weights))

    bar_width = 0.2  # Adjust this value to control the width of the bars
    plt.figure(figsize=(20,15))
    for group_id in range(num_groups):
        plt.bar(x + group_id * bar_width, average_errors[:, group_id], width=bar_width, label=f'Group {group_id + 1}')
    plt.xticks(x + (bar_width * (num_groups - 1) / 2), weights)
    plt.xlabel('Weights')
    plt.ylabel('Average Reward Error')
    plt.title('Average Reward Error Across Weights and Groups')
    plt.legend()
    neatplot.save_figure('weighted group DPO')

# Folder where YAML files are stored
folder_path = '/home/uceesr4/policy_optimization/log-weighted-dpo/pg'

# Seeds and weights range
seeds = range(2021, 2031)
weights = [round(i * 0.1, 1) for i in range(11)]

# Initialize lists to store data
all_weights = []

# Initialize 2D array to store data
num_groups = 2  # Assuming two groups
all_average_errors = np.zeros((len(weights), num_groups))

# Iterate over weights
for weight_idx, weight in enumerate(weights):
    if weight==0:
        weight_str = f"[{weight:.1f},{int(1-weight)}]"
    elif weight==1:
        weight_str = f"[{weight:.1f},{int(1-weight)}]"
    else:
        weight_str = f"[{weight:.1f},{1-weight:.1f}]"
    print(weight_str)
    # Initialize list to store errors for the current weight
    errors_for_weight = []

    # Iterate over seeds
    for seed in seeds:
        seed_weight_folder = os.path.join(folder_path, f"pg-{seed}-{weight_str}")
        file_path = os.path.join(seed_weight_folder, f"reward_error_dpo.yml")

        # Load YAML file and extract the error
        data = load_yml(file_path)
        error = data[20]

        # Accumulate error for the current seed
        all_average_errors[weight_idx, :] += np.array(error)

    # Calculate the average error for the current weight
    all_average_errors[weight_idx, :] /= len(seeds)

    # Store results for plotting
    all_weights.append(weight_str)

# Plotting
plot_grouped_bar_graph(all_weights, all_average_errors, num_groups)
