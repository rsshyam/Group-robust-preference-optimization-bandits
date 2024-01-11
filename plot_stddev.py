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


def plot_grouped_bar_graph(weights, average_errors, std_devs, num_groups):
    x = np.arange(len(weights))

    bar_width = 0.2  # Adjust this value to control the width of the bars
    plt.figure(figsize=(20, 15))

    for group_id in range(num_groups):
        plt.bar(
            x + group_id * bar_width,
            average_errors[:, group_id],
            yerr=std_devs[:, group_id],
            capsize=4,
            width=bar_width,
            label=f'Group {group_id + 1}'
        )

    plt.xticks(x + (bar_width * (num_groups - 1) / 2), weights, fontsize=22)
    plt.xlabel('Weights', fontsize=36)
    plt.ylabel('Average Reward Error', fontsize=36)
    plt.title('Average Reward Error Across Weights and Groups', fontsize=38)
    plt.legend(fontsize=34)
    
    neatplot.save_figure('weighted group DPO')

# Folder where YAML files are stored
#folder_path = '/home/uceesr4/policy_optimization/log-weighted-dpo/pg'
#folder_path='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/20240110195635/pg'
#folder_path='/home/uceesr4/policy_optimization/log-weighted-dpo/20240110180513/pg'
#folder_path='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/20240111111723/pg'
folder_path='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/20240111112941/pg'
# Seeds and weights range
seeds = range(2021, 2026)
weights = [round(i * 0.1, 1) for i in range(11)]

# Initialize lists to store data
all_weights = []

# Initialize 2D arrays to store data
num_groups = 2  # Assuming two groups
all_average_errors = np.zeros((len(weights), num_groups))
all_std_devs = np.zeros((len(weights), num_groups))

# Iterate over weights
for weight_idx, weight in enumerate(weights):
    if weight == 0:
        weight_str = f"[{weight:.1f},{int(1 - weight)}]"
    elif weight == 1:
        weight_str = f"[{weight:.1f},{int(1 - weight)}]"
    else:
        weight_str = f"[{weight:.1f},{1 - weight:.1f}]"
    print(weight_str)
    # Initialize lists to store errors for the current weight
    errors_for_weight = []

    # Iterate over seeds
    for seed in seeds:
        seed_weight_folder = os.path.join(folder_path, f"pg-{seed}-{weight_str}")
        file_path = os.path.join(seed_weight_folder, f"reward_error_dpo.yml")

        # Load YAML file and extract the error
        data = load_yml(file_path)
        error = data[20]

        # Store error for the current seed
        errors_for_weight.append(error)
    print(len(errors_for_weight),'length')
    remove_count = int(0.1 * len(errors_for_weight))
    if remove_count==0:
        remove_count=1
    # Convert the list to a NumPy array for easier calculations
    errors_for_weight = np.array(errors_for_weight)
    #Calculate the number of elements to remove (10% from each end)
   
    print(remove_count)
    # Sort the array in ascending order
    sorted_errors = np.sort(errors_for_weight, axis=0)

    # Remove the top and bottom 10% values
    trimmed_errors = sorted_errors[remove_count:-remove_count]
    print(trimmed_errors)
    # Calculate the average and standard deviation of the trimmed errors
    average_error_for_weight = np.mean(trimmed_errors, axis=0)
    std_dev_for_weight = np.std(trimmed_errors, axis=0)
    # Calculate the average error and standard deviation for the current weight
    #average_error_for_weight = np.mean(errors_for_weight, axis=0)
    #std_dev_for_weight = np.std(errors_for_weight, axis=0)

    # Store results for plotting
    all_average_errors[weight_idx, :] = average_error_for_weight
    all_std_devs[weight_idx, :] = std_dev_for_weight

    # Store weight string for plotting
    all_weights.append(weight_str)

# Plotting
plot_grouped_bar_graph(all_weights, all_average_errors, all_std_devs, num_groups)
