import wandb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from visualisations_utils_wandb_api import (
    download_runs,
    process_max_fields,
    process_runs,
    group_process_runs
    )

import os
import neatplot
neatplot.set_style()

if __name__ == "__main__":
    # Specify the parameters for your data collection
    entity = 'robust-rl-project'
    project = 'bandits_dpo'
    group = 'state_dim2action_num8group_num2pref_data_num250weights[0.2,0.8]feature_typeswappedeval_metricargmax'
    group1 = 'state_dim2action_num8group_num2pref_data_num210weights[0.2,0.8]feature_typeswappedeval_metricargmax'
    group2 = 'state_dim2action_num8group_num2pref_data_num180weights[0.2,0.8]feature_typeswappedeval_metricargmax'
    group3 = 'state_dim2action_num8group_num2pref_data_num150weights[0.2,0.8]feature_typeswappedeval_metricargmax'
    weights_array = np.array(group.split('weights[')[-1].split(']')[0].split(','), dtype=float)
    filters = {
        # Add additional filters if needed
    }

     # Define the filters
    filters_dict_rdpo = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 19000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'config.rdpo_weighted_batches' : True,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'State': 'finished',
        'config.rdpo_adj': '0',
        'config.deterministic': True
    }
    filters_dict_rdpo_unweighted = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 19000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'State' : 'finished',
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'config.rdpo_adj': '0',
        'config.deterministic': True
    }
    filters_dict_rdpo_unweighted_det_list_true_true_250 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 19000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'State' : 'finished',
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'config.rdpo_adj': '0',
        'config.deterministic_list': [True,True],
        'config.pref_data_num': 250
    }
    filters_dict_rdpo_unweighted_det_list_true_true_210 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 19000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group1,
        'State' : 'finished',
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'config.rdpo_adj': '0',
        'config.deterministic_list': [True,True],
        'config.pref_data_num': 210
    }
    filters_dict_rdpo_unweighted_det_list_true_true_180 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 19000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group2,
        'State' : 'finished',
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'config.rdpo_adj': '0',
        'config.deterministic_list': [True,True],
        'config.pref_data_num': 180

    }
    filters_dict_rdpo_unweighted_det_list_true_true_150 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 19000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group3,
        'State' : 'finished',
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'config.rdpo_adj': '0',
        'config.deterministic_list': [True,True],
        'config.pref_data_num': 150
    }
    filters_dict_rdpo_unweighted_deter_ratio_1 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 20000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'State' : 'finished',
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.deterministic_ratio': 1,
         'config.true_reward_params': [[1,3,1,3],[3,1,3,1]]
    }
    filters_dict_rdpo_unweighted_deter_ratio_0pt8 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 20000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'State' : 'finished',
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.deterministic_ratio': 0.8,
        'config.true_reward_params': [[1,3,1,3],[3,1,3,1]]
    }
    filters_dict_rdpo_unweighted_deter_ratio_0pt6 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 20000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'State' : 'finished',
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.deterministic_ratio': 0.6,
        'config.true_reward_params': [[1,3,1,3],[3,1,3,1]]
    }
    filters_dict_rdpo_unweighted_deter_ratio_0pt4 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 20000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'State' : 'finished',
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.deterministic_ratio': 0.4,
        'config.true_reward_params': [[1,3,1,3],[3,1,3,1]]
    }
    filters_dict_rdpo_unweighted_deter_ratio_0pt2 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 20000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'State' : 'finished',
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.deterministic_ratio': 0.2,
        'config.true_reward_params': [[1,3,1,3],[3,1,3,1]]
    }
    filters_dict_rdpo_unweighted_deter_ratio_0 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 20000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'State' : 'finished',
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.deterministic_ratio': 0,
        'config.true_reward_params': [[1,3,1,3],[3,1,3,1]],
        'config.deterministic_list': {'$nin': [[False,True]]}
    }
    filters_dict_random_deter_false = {
        'config.dpo_type': 'random',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 20000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'State' : 'finished',
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.deterministic_ratio': -1,
        'config.true_reward_params': [[1,3,1,3],[3,1,3,1]]
    }
    filters_dict_random_deter_true = {
        'config.dpo_type': 'random',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 19000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'State' : 'finished',
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'config.rdpo_adj': '0',
        'config.deterministic': True,
        'config.true_reward_params': [[1,3,1,3],[3,1,3,1]]
    }
    filters_dict_dpo = {
        'config.dpo_type': 'dpo',
        'config.dpo_num_iters': 19000,
        'config.ipo_grad_type': 'Regression',
        'group': group,
        'State': 'finished',
        'config.deterministic': True
    }
    filters_dict_imp_samp = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': True,
        'config.dpo_num_iters': 19000,#15000 for pref data 120
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'group': group,
        'State': 'finished',
        'config.deterministic': True
    }
   
    # Assume you have the necessary functions and libraries imported
    metrics_to_collect = ['grad_norm', 'train_loss', 'reward_err_1', 'reward_err_2', 'reward_param_1', 'reward_param_2', 'reward_param_3', 'reward_param_4','group_weight_1','group_weight_2','val_loss','train_group_loss_1','train_group_loss_2','val_group_loss_1','val_group_loss_2','hist_group_loss_1','hist_group_loss_2','max_val_grp_loss','max_train_grp_loss','max_reward_err','max_kl_dist']
    # List of filters_dict values
    #filters_dicts = [filters_dict_rdpo, filters_dict_dpo, filters_dict_imp_samp]
    #filters_dicts = [filters_dict_rdpo_unweighted,filters_dict_dpo,filters_dict_imp_samp,filters_dict_random_deter_true]
    #filters_dicts=[filters_dict_rdpo_unweighted_deter_ratio_1,filters_dict_rdpo_unweighted_deter_ratio_0pt6,filters_dict_rdpo_unweighted_deter_ratio_0,filters_dict_random_deter_false]
    filters_dicts=[filters_dict_rdpo_unweighted_det_list_true_true_250,filters_dict_rdpo_unweighted_det_list_true_true_210,filters_dict_rdpo_unweighted_det_list_true_true_180,filters_dict_rdpo_unweighted_det_list_true_true_150]
    # Initialize dictionaries to accumulate metrics data
    all_metrics_history = {metric: [] for metric in metrics_to_collect}
    all_runs=[]
    # Loop through each filters_dict value
    for filters_dict in filters_dicts:
        # Download runs for the current filters_dict
        runs = download_runs(entity, project, filters_dict)
        all_runs.append(runs)
        print(len(runs))
        metrics_history = {}

        for metric in metrics_to_collect:
            algo=filters_dict['config.dpo_type']
            det_ratio=filters_dict['config.deterministic_ratio']
            if algo=='dpo' and 'group_weight' in metric:
                metrics_history[metric] =[]
                continue
            metrics_history[metric] = process_runs(runs, field=metric, time_field='Iteration')

        # Accumulate metrics data for each configuration
        for metric in metrics_to_collect:
            all_metrics_history[metric].append(metrics_history[metric])
            
    iteration_len=0
    iteration_index=0
    for runs in all_runs:
        for run in runs:
            iteration_index_1=run[['Iteration']].dropna().values.ravel()
            print(iteration_index_1)
            if len(iteration_index_1)>iteration_len:
                iteration_len=len(iteration_index_1)
                iteration_index=iteration_index_1
 
    #print(metrics_history)



    # Specify the base folder path
    base_folder = f'wandb-plots/closed_form_large_data/{len(filters_dicts)}_{group}_deterministic_{str(filters_dicts[0]["config.deterministic_list"])}'

    # Create the base folder if it doesn't exist
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # Create a subfolder with keys and values
    #subfolder_name = '_'.join([f'{key}-{value}' for key, value in filters_dict[0].items()])
    subfolder_name=filters_dicts[0]['config.dpo_type']+str(filters_dicts[0]['config.dpo_num_iters'])+str(filters_dicts[0]['config.rdpo_weighted_batches'])
    subfolder_path = os.path.join(base_folder, subfolder_name)

    # Create the subfolder
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)


     # Plot all configurations for each metric in the same graph
    # Calculate average and standard error
    all_avg_metrics_at_iterations = {metric: [] for metric in metrics_to_collect}
    all_sem_metrics_at_iterations = {metric: [] for metric in metrics_to_collect}

    
    for i, filters_dict in enumerate(filters_dicts):
        for metric in metrics_to_collect:
            values_matrix = all_metrics_history[metric][i]
            #print(values_matrix)
            avg_values = np.mean(values_matrix, axis=0)
            sem_values = sem(all_metrics_history[metric][i], axis=0)
            all_avg_metrics_at_iterations[metric].append(avg_values.ravel())
            all_sem_metrics_at_iterations[metric].append(sem_values.ravel())

    
    # Plot metrics at 2000 iterations with error area
    #iteration_index = metrics_history['Iteration'][0]  # Assume iterations are the same for all runs
    print(iteration_index)
    plt.figure(figsize=(12, 6))

    for metric in ['train_loss','val_loss']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            det_ratio=filters_dict['config.deterministic_ratio']
            if algo=='rdpo' and filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            if 'd' in algo:
                algo=algo.replace('d','i')
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]
            print(len(avg_values),len(iteration_index),algo)
            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Train and Val Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/train_val')
    plt.close()

    plt.figure(figsize=(12, 6))

    for metric in ['grad_norm']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            det_ratio=filters_dict['config.deterministic_ratio']
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)
    plt.title('Grad Norm')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/grad')
    plt.close()

    
    plt.figure(figsize=(12, 6))
    for j,metric in enumerate(['reward_err_1', 'reward_err_2']):
        weight=str(weights_array[j])
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            det_ratio=filters_dict['config.deterministic_ratio']
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric+'_'+weight+'_'+ algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric+'_'+weight+'_'+ algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Group wise Reward Errors')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/reward_err')
    plt.close()


    plt.figure(figsize=(12, 6))
    for metric in ['max_reward_err']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            det_ratio=filters_dict['config.deterministic_ratio']
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Max Reward Errors')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_reward_err')
    plt.close()

    plt.figure(figsize=(12, 6))
    for metric in ['reward_param_1', 'reward_param_2']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            det_ratio=filters_dict['config.deterministic_ratio']
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)
    plt.title('Reward Parameters')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/reward_params_1_2')
    plt.close()

    plt.figure(figsize=(12, 6))
    for metric in ['reward_param_3', 'reward_param_4']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            det_ratio=filters_dict['config.deterministic_ratio']
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Reward Parameters')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/reward_params_3_4')
    plt.close()
    
    colors=['blue','orange']
    plt.figure(figsize=(12, 6))
    for j,metric in enumerate(['group_weight_1', 'group_weight_2']):
        weight=str(weights_array[j])
        color=colors[j]
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            det_ratio=filters_dict['config.deterministic_ratio']
            if algo=='dpo' and 'group_weight' in metric:
                continue
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values,color=color, label=metric+'_'+weight+'_'+ algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, color=color, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values,color=color, label=metric+'_'+weight+'_'+ algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values,color=color, alpha=0.2)

    plt.title('Group Weights')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/group_weights')
    plt.close()

    mult_runs=5
    colors=['blue','orange']
    rand_indices=np.random.choice(20,size=mult_runs)
    plt.figure(figsize=(12, 6))
    for j,metric in enumerate(['group_weight_1', 'group_weight_2']):
        weight=str(weights_array[j])
        color=colors[j]
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            det_ratio=filters_dict['config.deterministic_ratio']
            if algo=='dpo' and 'group_weight' in metric:
                continue
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            if 'ripo' in algo:
                for k in range(mult_runs):
                    rand_index=rand_indices[k]
                    avg_values = all_metrics_history[metric][i][rand_index]
                    if len(avg_values) != len(iteration_index):
                        # Extend avg_values by repeating the last value
                        extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                        plt.plot(iteration_index, extended_avg_values,color=color, label=metric+'_'+weight+'_'+ algo + '_det_ratio_' + str(det_ratio))
                    else:
                        plt.plot(iteration_index, avg_values,color=color, label=metric+'_'+weight+'_'+ algo + '_det_ratio_' + str(det_ratio))

            else:
                avg_values = all_avg_metrics_at_iterations[metric][i]
                sem_values = all_sem_metrics_at_iterations[metric][i]

                if len(avg_values) != len(iteration_index):
                    # Extend avg_values by repeating the last value
                    extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                    extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                    plt.plot(iteration_index, extended_avg_values, label=metric+'_'+weight+'_'+ algo + '_det_ratio_' + str(det_ratio))
                    plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
                else:
                    plt.plot(iteration_index, avg_values, label=metric+'_'+weight+'_'+ algo + '_det_ratio_' + str(det_ratio))
                    plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Group Weights Multiple Runs')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/group_weights_multiple_runs')
    plt.close()

    plt.figure(figsize=(12, 6))
    for j,metric in enumerate(['train_group_loss_1','train_group_loss_2']):
        weight=str(weights_array[j])
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            det_ratio=filters_dict['config.deterministic_ratio']
            #if algo=='dpo' and 'group_weight' in metric:
            #    continue
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric +'_ratio'+ weight +'_'+ algo+ '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric +'_ratio'+ weight +'_'+ algo+ '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Group Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/group_train_loss')
    plt.close()


    plt.figure(figsize=(12, 6))
    for metric in ['max_train_grp_loss']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            det_ratio=filters_dict['config.deterministic_ratio']
            #if algo=='dpo' and 'group_weight' in metric:
            #    continue
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)
    plt.title('Max train Group Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_group_train_loss')
    plt.close()

    plt.figure(figsize=(12, 6))
    for j,metric in enumerate(['hist_group_loss_1','hist_group_loss_2']):
        weight=str(weights_array[j])
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            det_ratio=filters_dict['config.deterministic_ratio']
            #if algo=='dpo' and 'group_weight' in metric:
            #    continue
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric +'_ratio'+ weight +'_'+ algo+ '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric +'_ratio'+ weight +'_'+ algo+ '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Group Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/hist_group_loss')
    plt.close()

    plt.figure(figsize=(12, 6))
    for j,metric in enumerate(['val_group_loss_1','val_group_loss_2']):
        weight=str(weights_array[j])
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            det_ratio=filters_dict['config.deterministic_ratio']
            #if algo=='dpo' and 'group_weight' in metric:
            #    continue
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric +'_ratio'+ weight +'_'+ algo+ '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric +'_ratio'+ weight +'_'+ algo+ '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Group Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/group_val_loss')
    plt.close()


    plt.figure(figsize=(12, 6))
    for metric in ['max_val_grp_loss']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            det_ratio=filters_dict['config.deterministic_ratio']
            #if algo=='dpo' and 'group_weight' in metric:
            #    continue
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Max Val Group Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_group_val_loss')
    plt.close()


    plt.figure(figsize=(12, 6))
    for metric in ['max_kl_dist']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            det_ratio=filters_dict['config.deterministic_ratio']
            #if algo=='dpo' and 'group_weight' in metric:
            #    continue
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo + '_det_ratio_' + str(det_ratio))
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)


    plt.title('Max KL Distance')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_kl_dist')
    plt.close()


    
    plt.figure(figsize=(12, 6))
    # Bar plot for reward_errs at the end
    for i, filters_dict in enumerate(filters_dicts):
        algo=filters_dict['config.dpo_type']
        det_ratio=filters_dict['config.deterministic_ratio']
        if algo=='rdpo':
            if filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            elif filters_dict['config.rdpo_weighted_batches']==True:
                algo='rdpo_weightbatch'
            else:
                algo='rdpo_unweightbatch'
        if 'd' in algo:
            algo=algo.replace('d','i')
        reward_errs_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'reward_err_' in metric]
        reward_errs_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'reward_err_' in metric]
        print(reward_errs_end_avg)
        plt.bar(np.arange(len(reward_errs_end_avg))+i*0.25, height= reward_errs_end_avg,yerr= reward_errs_end_sem,width=0.2,capsize=5,alpha=0.7,label=algo+ '_det_ratio_' + str(det_ratio))

        plt.xticks(np.arange(len(reward_errs_end_avg))+i*0.25, [f"Group {i+1}_ratio_{weights_array[i]}" for i in range(len(reward_errs_end_avg))])
    plt.title('Reward Errors at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/reward_bars')
    plt.close()

    plt.figure(figsize=(12, 6))
    # Bar plot for train_group_loss at the end
    for i, filters_dict in enumerate(filters_dicts):
        algo=filters_dict['config.dpo_type']
        det_ratio=filters_dict['config.deterministic_ratio']
        if algo=='rdpo':
            if filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            elif filters_dict['config.rdpo_weighted_batches']==True:
                algo='rdpo_weightbatch'
            else:
                algo='rdpo_unweightbatch'
        if 'd' in algo:
            algo=algo.replace('d','i')
        group_loss_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'train_group_loss' in metric]
        group_loss_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'train_group_loss' in metric]
        print(group_loss_end_avg)
        plt.bar(np.arange(len(group_loss_end_avg))+i*0.125, height= group_loss_end_avg,yerr= group_loss_end_sem,width=0.1,capsize=5,alpha=0.7,label=algo+ '_det_ratio_' + str(det_ratio))

        plt.xticks(np.arange(len(group_loss_end_avg))+i*0.125, [f"Group {i+1}_ratio_{weights_array[i]}" for i in range(len(group_loss_end_avg))])
    plt.title('Group Train Loss at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/train_group_loss_bars')
    plt.close()

    plt.figure(figsize=(12, 6))
    # Bar plot for val_group_loss at the end
    for i, filters_dict in enumerate(filters_dicts):
        algo=filters_dict['config.dpo_type']
        det_ratio=filters_dict['config.deterministic_ratio']
        if algo=='rdpo':
            if filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            elif filters_dict['config.rdpo_weighted_batches']==True:
                algo='rdpo_weightbatch'
            else:
                algo='rdpo_unweightbatch'
        if 'd' in algo:
            algo=algo.replace('d','i')
        group_loss_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'val_group_loss' in metric]
        group_loss_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'val_group_loss' in metric]
        print(group_loss_end_avg)
        plt.bar(np.arange(len(group_loss_end_avg))+i*0.125, height= group_loss_end_avg,yerr= group_loss_end_sem,width=0.1,capsize=5,alpha=0.7,label=algo+ '_det_ratio_' + str(det_ratio))

        plt.xticks(np.arange(len(group_loss_end_avg))+i*0.125, [f"Group {i+1}_ratio_{weights_array[i]}" for i in range(len(group_loss_end_avg))])
    plt.title('Group Validation Loss at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/val_group_loss_bars')
    plt.close()
     
    #'max_val_grp_loss','max_train_grp_loss','max_reward_err','max_kl_dist'
    plt.figure(figsize=(12, 6))
    # Bar plot for reward_errs at the end
    for i, filters_dict in enumerate(filters_dicts):
        algo=filters_dict['config.dpo_type']
        det_ratio=filters_dict['config.deterministic_ratio']
        if algo=='rdpo':
            if filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            elif filters_dict['config.rdpo_weighted_batches']==True:
                algo='rdpo_weightbatch'
            else:
                algo='rdpo_unweightbatch'
        if 'd' in algo:
            algo=algo.replace('d','i')
        max_reward_err_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_reward_err' in metric]
        max_reward_err_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_reward_err' in metric]
        print(max_reward_err_end_avg)
        plt.bar(np.arange(len(max_reward_err_end_avg))+i*0.25, height= max_reward_err_end_avg,yerr= max_reward_err_end_sem,width=0.2,capsize=5,alpha=0.7,label=algo+ '_det_ratio_' + str(det_ratio))

        #plt.xticks(np.arange(len(max_reward_err_end_avg))+i*0.25, [f"Group {i+1}" for i in range(len(max_reward_err_end_avg))])
    plt.title('Max Reward Error at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_reward_bars')
    plt.close()

    plt.figure(figsize=(12, 6))
    # Bar plot for train_group_loss at the end
    for i, filters_dict in enumerate(filters_dicts):
        algo=filters_dict['config.dpo_type']
        det_ratio=filters_dict['config.deterministic_ratio']
        if algo=='rdpo':
            if filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            elif filters_dict['config.rdpo_weighted_batches']==True:
                algo='rdpo_weightbatch'
            else:
                algo='rdpo_unweightbatch'
        if 'd' in algo:
            algo=algo.replace('d','i')
        max_group_loss_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_train_grp_loss' in metric]
        max_group_loss_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_train_grp_loss' in metric]
        print(max_group_loss_end_avg)
        plt.bar(np.arange(len(max_group_loss_end_avg))+i*0.125, height= max_group_loss_end_avg,yerr= max_group_loss_end_sem,width=0.1,capsize=5,alpha=0.7,label=algo+ '_det_ratio_' + str(det_ratio))

        #plt.xticks(np.arange(len(group_loss_end_avg))+i*0.125, [f"Group {i+1}" for i in range(len(group_loss_end_avg))])
    plt.title('Max Group Train Loss at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_train_group_loss_bars')
    plt.close()

    plt.figure(figsize=(12, 6))
    # Bar plot for val_group_loss at the end
    for i, filters_dict in enumerate(filters_dicts):
        algo=filters_dict['config.dpo_type']
        det_ratio=filters_dict['config.deterministic_ratio']
        if algo=='rdpo':
            if filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            elif filters_dict['config.rdpo_weighted_batches']==True:
                algo='rdpo_weightbatch'
            else:
                algo='rdpo_unweightbatch'
        if 'd' in algo:
            algo=algo.replace('d','i')
        max_group_loss_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_val_grp_loss' in metric]
        max_group_loss_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_val_grp_loss' in metric]
        print(max_group_loss_end_avg)
        plt.bar(np.arange(len(max_group_loss_end_avg))+i*0.125, height= max_group_loss_end_avg,yerr= max_group_loss_end_sem,width=0.1,capsize=5,alpha=0.7,label=algo+ '_det_ratio_' + str(det_ratio))

        #plt.xticks(np.arange(len(max_group_loss_end_avg))+i*0.125, [f"Group {i+1}" for i in range(len(group_loss_end_avg))])
    plt.title('Max Group Validation Loss at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_val_group_loss_bars')
    plt.close()

    plt.figure(figsize=(12, 6))
    # Bar plot for val_group_loss at the end
    for i, filters_dict in enumerate(filters_dicts):
        algo=filters_dict['config.dpo_type']
        det_ratio=filters_dict['config.deterministic_ratio']
        if algo=='rdpo':
            if filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            elif filters_dict['config.rdpo_weighted_batches']==True:
                algo='rdpo_weightbatch'
            else:
                algo='rdpo_unweightbatch'
        if 'd' in algo:
            algo=algo.replace('d','i')
        max_kl_dist_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_kl_dist' in metric]
        max_kl_dist_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_kl_dist' in metric]
        print(max_kl_dist_end_avg)
        plt.bar(np.arange(len(max_kl_dist_end_avg))+i*0.125, height= max_kl_dist_end_avg,yerr= max_kl_dist_end_sem,width=0.1,capsize=5,alpha=0.7,label=algo+ '_det_ratio_' + str(det_ratio))

        #plt.xticks(np.arange(len(max_group_loss_end_avg))+i*0.125, [f"Group {i+1}" for i in range(len(group_loss_end_avg))])
    plt.title('Max KL distance at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_kl_distance_bars')
    plt.close()
