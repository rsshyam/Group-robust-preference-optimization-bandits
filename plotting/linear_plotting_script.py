# -*- coding: utf-8 -*-
"""
Created on [Anonymised]

@author: [Anonymised]
"""

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import neatplot
import re

neatplot.set_style()

exp_id = 1
check_weights=True

log_name = "pg"
log_folder_path='/[Anonymised]'
match = re.search(r'sep/(dpo|rdpo)/2024', log_folder_path)
if match.group(1)=='dpo':
    check_weights=False

# Function to extract pref_data_num from config.yaml
def extract_config_field(config_path, field):
    with open(config_path, 'r') as config_file:
        config_data = yaml.safe_load(config_file)
        return config_data.get(field)
    
def extract_seq_error_data(log_path):
    with open(log_path, 'r') as log_file:
        lines = log_file.readlines()
        outputs = list()
        one_iter = list()
        for line in lines:
            if ("linear_bandit/group_dpo.py" in line or "linear_bandit/group_robust_dpo.py" in line) and ("Iteration" in line):
                results = line.split("INFO: Iteration:")[-1].split(' ')
                #print('seq_error_data', results)
                epoch = float(results[2][:-1]) #get rid of comma
                val_loss = float(results[8][:-1]) #TODO: will need to get rid of comma
                train_loss   = float(results[5][:-1])
                one_iter.append([epoch, train_loss, val_loss])
            if ("INFO: Policy parameter learned solely on the preference data" in line):
                outputs.append(one_iter)
                one_iter = list()
    return outputs

def extract_seq_error_data_loss(log_path):
    with open(log_path, 'r') as log_file:
        lines = log_file.readlines()
        outputs = list()
        one_iter = list()
        for line in lines:
            if ("linear_bandit/group_dpo.py" in line or "linear_bandit/group_robust_dpo.py" in line) and ("Iteration" in line):
                results = line.split("INFO: Iteration:")[-1].split(' ')
                #print('seq_error_data', results)
                epoch = float(results[2].split(',')[0]) #get rid of comma
                train_loss   = float(results[5].split(',')[0])
                val_loss = float(results[8].split(',')[0])
                one_iter.append([epoch, train_loss, val_loss])
                #print([epoch, train_loss, val_loss])
            if ("INFO: Policy parameter learned solely on the preference data" in line):
                outputs.append(one_iter)
                one_iter = list()
    return outputs

def extract_seq_error_data_grad(log_path):
    with open(log_path, 'r') as log_file:
        lines = log_file.readlines()
        outputs = list()
        one_iter = list()
        for line in lines:
            if ("linear_bandit/group_dpo.py" in line or "linear_bandit/group_robust_dpo.py" in line) and ("Iteration" in line):
                results = line.split("INFO: Iteration:")[-1].split(' ')
                #print('seq_error_data', results)
                epoch = float(results[2].split(',')[0]) #get rid of comma
                grad_norm = float(results[10].split(',')[0])
                one_iter.append([epoch, grad_norm])
                #print([epoch, grad_norm])
                #raise LookupError
                #print([epoch, train_loss, val_loss])
            if ("INFO: Policy parameter learned solely on the preference data" in line):
                outputs.append(one_iter)
                one_iter = list()
    return outputs

def extract_seq_error_data_rewards(log_path):
    with open(log_path, 'r') as log_file:
        lines = log_file.readlines()
        outputs = list()
        one_iter = list()
        for line in lines:
            if ("linear_bandit/group_dpo.py" in line or "linear_bandit/group_robust_dpo.py" in line) and ("Iteration" in line):
                results = line.split("INFO: Iteration:")[-1].split(' ')
                #print('seq_error_data', results)
                #print('seq_error_data', len(results))
                epoch = float(results[2].split(',')[0]) #get rid of comma
                rewards_2   = float(results[13].split(',')[0])
                rewards_1   = float(results[12].split(',')[0])
                #val_loss = float(results[8].split(',')[0])
                one_iter.append([epoch, rewards_1,rewards_2])
                #print(rewards_1,rewards_2)
                #raise LookupError
                #print([epoch, train_loss, val_loss])
            if ("INFO: Policy parameter learned solely on the preference data" in line):
                outputs.append(one_iter)
                one_iter = list()
    return outputs

def extract_seq_error_data_weights(log_path):
    with open(log_path, 'r') as log_file:
        lines = log_file.readlines()
        outputs = list()
        one_iter = list()
        for line in lines:
            if ("linear_bandit/group_robust_dpo.py" in line) and ("Iteration" in line):
                results = line.split("INFO: Iteration:")[-1].split(' ')
                #print('seq_error_data', results)
                epoch = float(results[2].split(',')[0]) #get rid of comma
                train_loss   = float(results[5].split(',')[0])
                #print(results[15].split('['))
                weights_1 = float(results[15].split('[')[-1])
                weights_2 = float(results[16].split(']')[0])
                #print(weights_1,weights_2)
                one_iter.append([weights_1, weights_2])
            if ("INFO: Policy parameter learned solely on the preference data" in line):
                outputs.append(one_iter)
                one_iter = list()
    return outputs


def extract_dpo_reward_error(log_path):
    with open(log_path, 'r') as log_file:
        lines = log_file.readlines()
        print(lines)
        outputs = list()
        for i, line in enumerate(lines):
            if "INFO: Number of selected points: " in line:
                num_points = float(line.split("INFO: Number of selected points: ")[-1].strip())
                error = float(lines[i+1].split("INFO: DPO reward error: ")[-1].strip())
                outputs.append((num_points, error))
        return outputs
                
# Function to extract last DPO reward error from log.txt
def extract_last_dpo_reward_error(log_path):
    with open(log_path, 'r') as log_file:
        lines = log_file.readlines()
        for line in reversed(lines):
            if "reward error: " in line:
                #print(line.split("reward error: ")[-1])
                #print(line.split("reward error: ")[-1].strip())
                value_str=line.split("reward error: ")[-1].strip()
                if not value_str[-1].isdigit():
                    value_str=value_str[:-1]
                #numeric_value_str = ''.join(char for char in value_str if char.isdigit() or char == '.' or char == '-')
                #print(value_str)
                values = [float(value.strip()) for value in value_str.split(',')]
                return values
                #return float(line.split("reward error: ")[-1].strip())

# Dictionary to store pref_data_num and corresponding DPO reward errors
experiments = dict()

# Iterate through folders in log/pg
for folder_name in os.listdir(log_folder_path):
    folder_path = os.path.join(log_folder_path, folder_name)
    #print(folder_path)
    # Check if it's a directory and starts with "pg"
    if os.path.isdir(folder_path) and folder_name.startswith(log_name):
        #print('hello')
        # Construct paths to config.yaml and log.txt
        config_path = os.path.join(folder_path, "config.yaml")
        log_path = os.path.join(folder_path, "log.txt")
        
        final_error_path=os.path.join(folder_path, f"reward_error_{match.group(1)}.yml")

        if not os.path.isfile(final_error_path):
            continue
        # Extract exp_id
        config_id = extract_config_field(config_path, 'weights')+'_'+str(extract_config_field(config_path, 'seed'))
        #print(config_id,'config_id')
        # Extract per experiment sequential data:
        per_exp_seq_data = extract_seq_error_data_loss(log_path)

        per_exp_seq_data_grad = extract_seq_error_data_grad(log_path)
        #print(per_exp_seq_data)
        #Extract last DPO reward error
        #last_dpo_reward_error = extract_last_dpo_reward_error(log_path)
        #print(last_dpo_reward_error)         
        if check_weights:
            per_exp_weights=extract_seq_error_data_weights(log_path)
        per_exp_rewards = extract_seq_error_data_rewards(log_path)
        #print(per_exp_rewards)
        # Update dictionary with pref_data_num and corresponding errors
        if config_id not in experiments:
            experiments[config_id] = dict()
            #experiments[config_id]['seq_data'] = list()
            experiments[config_id]['per_exp_seq_data'] = list()
            experiments[config_id]['per_exp_seq_data_grad'] = list()
            #experiments[config_id]['last_dpo_reward_error'] = list()
            if check_weights:
                experiments[config_id]['per_exp_weights']=list()
            experiments[config_id]['per_exp_rewards'] = list()
        #experiments[config_id]['seq_data'].append(seq_data)
        experiments[config_id]['per_exp_seq_data'].append(per_exp_seq_data)
        experiments[config_id]['per_exp_seq_data_grad'].append(per_exp_seq_data_grad)
        #experiments[config_id]['last_dpo_reward_error'].append(last_dpo_reward_error)
        if check_weights:
            experiments[config_id]['per_exp_weights'].append(per_exp_weights)
        experiments[config_id]['per_exp_rewards'].append(per_exp_rewards)
#print(experiments)
def plot_experiment(experiments, exp_id, label, color, fig_axs=None, xlim=[0,50]):

    #Process experiments into dataframe:
    exps = [pd.DataFrame(exper, columns=['NumPoints', 'ErrorGap']).set_index('NumPoints')\
            for exper in experiments[exp_id]['last_dpo_reward_error']]
        
    #Calculate mean and standard deviation across the same experiment number runs
    exps = pd.concat(exps, axis=1)
    mean = exps.mean(axis=1)
    std  = exps.std(axis=1)
    
    #Plot the results
    if fig_axs is None:
        fig, axs = plt.subplots()
    else:
        fig, axs = fig_axs
        
    #Remove highest and lowest values:
        
    axs.set_title('Num of Points vs Suboptimality Gap')
    axs.set_xlabel('Num of Points')
    axs.set_ylabel('Suboptimality Gap')
    axs.set_xlim(xlim)
    
    axs.plot(exps.index, mean, color=color, label=label)
    
    if exps.shape[-1] > 1:
        axs.fill_between(exps.index, mean-std, mean+std, alpha=0.2, color=color)
    plt.legend()
    
    return fig, axs

#print('experiment keys', experiments.keys())
prefixes=set(key.split('_')[0] for key in experiments.keys())
#print(prefixes)
save_dir=f'{log_folder_path.split("/pg")[0]}/allplots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for prefix in prefixes:
    keys_with_prefix = [key for key in experiments.keys() if key.startswith(prefix)]
    #steps = len(experiments[keys_with_prefix[0]])  # Assuming all keys have the same number of steps
    #print('hello')
    # Collecting weights for each key with the same prefix
    if check_weights:

        all_weights = np.array([list(experiments[key]['per_exp_weights']) for key in keys_with_prefix],  dtype=np.float64).squeeze()
        #print(all_weights.shape)
        #print(np.mean(all_weights,axis=0).shape)
        # Calculating mean and std across the weights for each training step
        mean_weights = np.mean(all_weights, axis=0)
        std_weights = np.std(all_weights, axis=0)
        
        fig, axs = plt.subplots()
        # Plotting mean curve
        steps=mean_weights.shape[0]
        x_range = [x*20 for x in range(1, steps + 1)]
        axs.plot(x_range, mean_weights[:,0], label=f'{prefix} Ratio - Group Weights-1')
        
        # Plotting std curve as shaded region
        axs.fill_between(x_range, mean_weights[:,0] - std_weights[:,0], mean_weights[:,0] + std_weights[:,0], alpha=0.2)

        # Plotting mean curve
        axs.plot(x_range, mean_weights[:,1], label=f'{prefix} Ratio - Group Weights-2')
        
        # Plotting std curve as shaded region
        axs.fill_between(x_range, mean_weights[:,1] - std_weights[:,1], mean_weights[:,1] + std_weights[:,1], alpha=0.2)


        
        axs.set_xlabel('Epoch')
        axs.set_ylabel('Weights')
        axs.set_title(f'Weights for RDPO')
        
        axs.legend()
        neatplot.save_figure(f'{save_dir}/group_weights_for_{prefix}')
    
    #print([list(experiments[key]['per_exp_seq_data']) for key in keys_with_prefix])
    # Collecting weights for each key with the same prefix
    epoch_train_val_loss = np.array([list(experiments[key]['per_exp_seq_data']) for key in keys_with_prefix],  dtype=np.float64).squeeze()
    #print(epoch_train_val_loss.shape)
    #print(np.mean(epoch_train_val_loss,axis=0).shape)
    # Calculating mean and std across the weights for each training step
    mean_losses = np.mean(epoch_train_val_loss, axis=0)
    std_losses = np.std(epoch_train_val_loss, axis=0)
    #print(mean_losses)
    fig, axs = plt.subplots()
    # Plotting mean curve
    #steps=mean_weights.shape[0]
    #x_range = [x*20 for x in range(1, steps + 1)]
    axs.plot(mean_losses[:,0], mean_losses[:,1], label=f'{prefix} Train Loss')
    
    # Plotting std curve as shaded region
    axs.fill_between(mean_losses[:,0], mean_losses[:,1]- std_losses[:,1], mean_losses[:,1]+ std_losses[:,1], alpha=0.2)

    axs.plot(mean_losses[:,0], mean_losses[:,2], label=f'{prefix} Validation Loss')
    
    # Plotting std curve as shaded region
    axs.fill_between(mean_losses[:,0], mean_losses[:,2]- std_losses[:,2], mean_losses[:,2]+ std_losses[:,2], alpha=0.2)

    
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Loss')
    axs.set_title(f'Loss for {match.group(1)}')
    
    axs.legend()
    neatplot.save_figure(f'{save_dir}/loss_for_{prefix}')

    
    
    #print([list(experiments[key]['per_exp_seq_data']) for key in keys_with_prefix])
    # Collecting weights for each key with the same prefix
    epoch_train_grad = np.array([list(experiments[key]['per_exp_seq_data_grad']) for key in keys_with_prefix],  dtype=np.float64).squeeze()
    #print(epoch_train_val_loss.shape)
    #print(np.mean(epoch_train_val_loss,axis=0).shape)
    # Calculating mean and std across the weights for each training step
    mean_grad = np.mean(epoch_train_grad, axis=0)
    std_grad = np.std(epoch_train_grad, axis=0)
    #print(mean_grad)
    fig, axs = plt.subplots()
    # Plotting mean curve
    #steps=mean_weights.shape[0]
    #x_range = [x*20 for x in range(1, steps + 1)]
    axs.plot(mean_grad[:,0], mean_grad[:,1], label=f'{prefix} Train Grad Norm')
    
    # Plotting std curve as shaded region
    axs.fill_between(mean_grad[:,0], mean_grad[:,1]- std_grad[:,1], mean_grad[:,1]+ std_grad[:,1], alpha=0.2)
    
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Grad Norm')
    axs.set_title(f'Grad Norm for {match.group(1)}')
    
    axs.legend()
    neatplot.save_figure(f'{save_dir}/Grad_for_{prefix}')

    #print([list(experiments[key]['per_exp_seq_data']) for key in keys_with_prefix])
    # Collecting weights for each key with the same prefix
    epoch_rewards = np.array([list(experiments[key]['per_exp_rewards']) for key in keys_with_prefix],  dtype=np.float64).squeeze()
    #print(epoch_train_val_loss.shape)
    #print(np.mean(epoch_train_val_loss,axis=0).shape)
    # Calculating mean and std across the weights for each training step
    mean_rewards = np.mean(epoch_rewards, axis=0)
    std_rewards = np.std(epoch_rewards, axis=0)
    print(mean_rewards)
    fig, axs = plt.subplots()
    # Plotting mean curve
    #steps=mean_weights.shape[0]
    #x_range = [x*20 for x in range(1, steps + 1)]
    axs.plot(mean_rewards[:,0], mean_rewards[:,1], label=f'{prefix} Rewards-1')
    
    # Plotting std curve as shaded region
    axs.fill_between(mean_rewards[:,0], mean_rewards[:,1]- std_rewards[:,1], mean_rewards[:,1]+ std_rewards[:,1], alpha=0.2)

    axs.plot(mean_rewards[:,0], mean_rewards[:,2], label=f'{prefix} Rewards-2')
    
    # Plotting std curve as shaded region
    axs.fill_between(mean_rewards[:,0], mean_rewards[:,2]- std_rewards[:,2], mean_rewards[:,2]+ std_rewards[:,2], alpha=0.2)

    
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Rewards')
    axs.set_title(f'Rewards for {match.group(1)}')
    
    axs.legend()
    neatplot.save_figure(f'{save_dir}/Rewards_for_{prefix}')




#%%

#figa = plot_experiment(experiments, exp_id= list(experiments.keys())[0], label='random', color='tab:blue', xlim=[15,70])
#plot_experiment(experiments, exp_id=list(experiments.keys())[1], label='ae_dpo', color='tab:orange', xlim=[15,70], fig_axs=figa)
 
#%% New process idea:
#import numpy as np
    
#exp_id_list = list(experiments.keys())
#alg = 'Random'

#exps = np.array([experiments[exp_id]['per_exp_weights'] for exp_id in exp_id_list])
#print(exps.shape)
#Data 1 -> for each of the 12 different data sizes across 5 seeds plot results:

#num_data_sizes = exps.shape[1]
"""
data_sizes = list(range(15,75,5))
train_index = 1
val_index = 2

for i, s in enumerate(data_sizes):
    
    index = exps[0,i,:,0]
    mean = exps[:,i,:,train_index].mean(axis=0)
    std  = exps[:,i,:,train_index].std(axis=0)

    fig, axs = plt.subplots()
    
    axs.plot(index, mean, label='Train Loss')
    axs.fill_between(index, mean-std, mean+std, alpha=0.2, color='tab:blue')
    
    mean = exps[:,i,:,val_index].mean(axis=0)
    std  = exps[:,i,:,val_index].std(axis=0)
    
    axs.plot(index, mean, label='Val Loss', color='tab:orange')
    axs.fill_between(index, mean-std, mean+std, alpha=0.2, color='tab:orange')
    
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Loss')
    axs.set_title(f'Losses for Dataset Size {data_sizes[i]}, {alg} Af')
    
    plt.legend()
"""