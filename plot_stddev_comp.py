import neatplot
neatplot.set_style()

import os
import yaml
import matplotlib.pyplot as plt
import numpy as np

dta_size=40
remove_outliers_bool=False

def load_yml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def retrieve_data(folder_path,seed,weight_str,loss):
    seed_weight_folder = os.path.join(folder_path, f"pg-{seed}-{weight_str}")
    file_path = os.path.join(seed_weight_folder, f"reward_error_{loss}.yml")

        # Load YAML file and extract the error
    data = load_yml(file_path)
    error = data[dta_size]

    return error
        

def remove_outliers(errors_for_weight):
    remove_count = int(0.05 * len(errors_for_weight))
    if remove_count==0:
        remove_count=1
    # Convert the list to a NumPy array for easier calculations
    errors_for_weight = np.array(errors_for_weight)
    sorted_errors = np.sort(errors_for_weight, axis=0)

    # Remove the top and bottom 10% values
    trimmed_errors = sorted_errors[remove_count:-remove_count]
    print(trimmed_errors)
    print(sorted_errors[:remove_count],sorted_errors[-remove_count:],'outliers')
    return trimmed_errors

def plot_grouped_bar_graph(weights, average_errors_rdpo, std_devs_rdpo, average_errors_dpo, std_devs_dpo, num_groups):
    x = np.arange(len(weights))

    bar_width = 0.1  # Adjust this value to control the width of the bars
    plt.figure(figsize=(20, 15))

    for group_id in range(num_groups):
        plt.bar(
            x + group_id * bar_width,
            average_errors_rdpo[:, group_id],
            yerr=std_devs_rdpo[:, group_id],
            capsize=4,
            width=bar_width,
            label=f'Group {group_id + 1}-RDPO'
        )

    for group_id in range(num_groups):
        plt.bar(
            x+0.25 + group_id * bar_width,
            average_errors_dpo[:, group_id],
            yerr=std_devs_dpo[:, group_id],
            capsize=4,
            width=bar_width,
            label=f'Group {group_id + 1}-DPO'
        )

    plt.xticks(x + (bar_width * (num_groups - 1) / 2), weights, fontsize=22)
    plt.xlabel('Data Ratios', fontsize=36)
    plt.ylabel('Average Reward Error', fontsize=36)
    plt.title('Average Reward Error Across Weights and Groups', fontsize=38)
    plt.legend(fontsize=34)
    
    neatplot.save_figure('weighted group RDPO-DPO Comparison')

# Folder where YAML files are stored
#folder_path = '/home/uceesr4/policy_optimization/log-weighted-dpo/pg'
#folder_path='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/20240110195635/pg'
#folder_path='/home/uceesr4/policy_optimization/log-weighted-dpo/20240110180513/pg'
#folder_path='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/20240111111723/pg'
#folder_path='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/20240111112941/pg'
#folder_path='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/20240111155055/pg'
#folder_path='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/20240111154217/pg'
#folder_path_rdpo='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/20240112131734/pg'
#folder_path_dpo='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/20240112120558/pg'
#folder_path_rdpo='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/rdpo/2024_01_15_04_22_16/pg'
#folder_path_dpo='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/dpo/2024_01_15_04_28_41/pg'

#folder_path_rdpo='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/rdpo/2024_01_15_21_53_52_200/pg'
#folder_path_dpo='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/dpo/2024_01_15_21_48_49_200/pg'

#folder_path_rdpo='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/rdpo/2024_01_16_02_10_47_200/pg'#step-size 0.5 non-adaptive
folder_path_rdpo='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/rdpo/2024_01_16_02_10_15_200/pg'#step-size 1 non-adaptive

#folder_path_dpo='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/dpo/2024_01_16_01_20_21_200/pg'#step-size 0.5 non-adaptive
folder_path_dpo='/home/uceesr4/policy_optimization/log-weighted-dpo_sep/dpo/2024_01_16_01_36_15_200/pg'#step-size 1 non-adaptive

# Seeds and weights range
seeds = range(2021, 2041)
weights = [round(i * 0.1, 1) for i in range(1,10)]

# Initialize lists to store data
all_weights = []

# Initialize 2D arrays to store data
num_groups = 2  # Assuming two groups
all_average_errors_rdpo = np.zeros((len(weights), num_groups))
all_std_devs_rdpo = np.zeros((len(weights), num_groups))

all_average_errors_dpo = np.zeros((len(weights), num_groups))
all_std_devs_dpo = np.zeros((len(weights), num_groups))

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
    errors_for_weight_rdpo = []
    errors_for_weight_dpo = []
    # Iterate over seeds
    for seed in seeds:
        error=retrieve_data(folder_path_rdpo,seed,weight_str,'rdpo')
        # Store error for the current seed
        errors_for_weight_rdpo.append(error)
        
        error=retrieve_data(folder_path_dpo,seed,weight_str,'dpo')
        # Store error for the current seed
        errors_for_weight_dpo.append(error)

    if remove_outliers_bool==True: 
        errors_for_weight_rdpo=np.array(errors_for_weight_rdpo)
        trimmed_errors_rdpo=remove_outliers(errors_for_weight_rdpo)
        
        # Calculate the average and standard deviation of the trimmed errors
        average_error_for_weight_rdpo = np.mean(trimmed_errors_rdpo, axis=0)
        std_dev_for_weight_rdpo = np.std(trimmed_errors_rdpo, axis=0)

        errors_for_weight_dpo=np.array(errors_for_weight_dpo)
        trimmed_errors_dpo=remove_outliers(errors_for_weight_dpo)
        
        # Calculate the average and standard deviation of the trimmed errors
        average_error_for_weight_dpo = np.mean(trimmed_errors_dpo, axis=0)
        std_dev_for_weight_dpo = np.std(trimmed_errors_dpo, axis=0)
    else:
        errors_for_weight_rdpo=np.array(errors_for_weight_rdpo)
        # Calculate the average error and standard deviation for the current weight
        average_error_for_weight_rdpo = np.mean(errors_for_weight_rdpo, axis=0)
        std_dev_for_weight_rdpo = np.std(errors_for_weight_rdpo, axis=0)

        errors_for_weight_dpo=np.array(errors_for_weight_dpo)
        # Calculate the average error and standard deviation for the current weight
        average_error_for_weight_dpo = np.mean(errors_for_weight_dpo, axis=0)
        std_dev_for_weight_dpo = np.std(errors_for_weight_dpo, axis=0)

    # Store results for plotting
    all_average_errors_rdpo[weight_idx, :] = average_error_for_weight_rdpo
    all_std_devs_rdpo[weight_idx, :] = std_dev_for_weight_rdpo

    # Store results for plotting
    all_average_errors_dpo[weight_idx, :] = average_error_for_weight_dpo
    all_std_devs_dpo[weight_idx, :] = std_dev_for_weight_dpo

    # Store weight string for plotting
    all_weights.append(weight_str)

# Plotting
plot_grouped_bar_graph(all_weights, all_average_errors_rdpo, all_std_devs_rdpo, all_average_errors_dpo, all_std_devs_dpo, num_groups)
