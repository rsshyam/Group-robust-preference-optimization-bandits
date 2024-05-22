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


def prune_bad_runs(values_matrix,n_remove):
    print(values_matrix)
    return values_matrix











if __name__ == "__main__":
    # Specify the parameters for your data collection
    entity = 'robust-rl-project'
    project = 'bandits_dpo'
    group = 'state_dim2action_num8group_num2pref_data_num60weights[0.2,0.8]feature_typeflippedeval_metricargmax'
    weights_array = np.array(group.split('weights[')[-1].split(']')[0].split(','), dtype=float)
    filters = {
        # Add additional filters if needed
    }

    # Define the filters
    filters_dict_rdpo_lamba_1 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 19000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'State': 'finished',
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.reg_coef': 0.1,
        'config.lamba': 1,
    }

    filters_dict_rdpo_lamba_1_wo_2_extreme = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 19000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'State': 'finished',
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.reg_coef': 0.1,
        'config.lamba': 1,
        'config.seed': {'$nin': [2025, 2030]}
    }

    filters_dict_rdpo_lamba_1_wo_4_extreme = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 19000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'State': 'finished',
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.reg_coef': 0.1,
        'config.lamba': 1,
        'config.seed': {'$nin': [2025, 2030, 2032,2027]}
    }


    filters_dict_imp_samp_lamba_1 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': True,
        'config.dpo_num_iters': 19000,
        'group': group,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'State': 'finished',
        'config.deterministic': False,
        'config.lamba': 1,
    }

    
    filters_dict_imp_samp_lamba_1_wo_2_extreme = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': True,
        'config.dpo_num_iters': 19000,
        'group': group,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'State': 'finished',
        'config.deterministic': False,
        'config.lamba': 1,
        'config.seed': {'$nin': [2031, 2033]}
    }

    filters_dict_imp_samp_lamba_1_wo_4_extreme = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': True,
        'config.dpo_num_iters': 19000,
        'group': group,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'State': 'finished',
        'config.deterministic': False,
        'config.lamba': 1,
        'config.seed': {'$nin': [2031, 2033, 2035, 2027]}
    }

    filters_dict_dpo_lamba_1 = {
        'config.dpo_type': 'dpo',
        'config.ipo_grad_type': 'Regression',
        'config.dpo_num_iters': 19000,
        'group': group,
        'State': 'finished',
        'config.deterministic': False,
        'config.reg_coef': 0.1,
        'config.lamba': 1
    }


    filters_dict_dpo_lamba_1_wo_2_extreme = {
        'config.dpo_type': 'dpo',
        'config.ipo_grad_type': 'Regression',
        'config.dpo_num_iters': 19000,
        'group': group,
        'State': 'finished',
        'config.deterministic': False,
        'config.reg_coef': 0.1,
        'config.lamba': 1,
        'config.seed': {'$nin': [2023, 2037]}
    }

    filters_dict_dpo_lamba_1_wo_4_extreme = {
        'config.dpo_type': 'dpo',
        'config.ipo_grad_type': 'Regression',
        'config.dpo_num_iters': 19000,
        'group': group,
        'State': 'finished',
        'config.deterministic': False,
        'config.reg_coef': 0.1,
        'config.lamba': 1,
        'config.seed': {'$nin': [2023, 2026, 2037, 2038]}
    }

    
    filters_dict_rdpo_lamba_1_25_32 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 15000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'State': 'finished',
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.reg_coef': 0.1,
        'config.lamba': 1,
        'config.seed': {'$in': [2025, 2032]}
    }

    filters_dict_rdpo_lamba_1_wo_32 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 15000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'State': 'finished',
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.reg_coef': 0.1,
        'config.lamba': 1,
        'config.seed': {'$nin': [2032]}
    }

    filters_dict_rdpo_lamba_1_32 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 15000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'State': 'finished',
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.reg_coef': 0.1,
        'config.lamba': 1,
        'config.seed': {'$in': [2032]}
    }
    
    filters_dict_rdpo_lamba_2 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 15000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'State': 'finished',
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.reg_coef': 0.1,
        'config.lamba': 2
    }
    filters_dict_rdpo_lamba_0 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 15000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'State': 'finished',
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.reg_coef': 0.1,
        'config.lamba': 0#0
    }
    filters_dict_rdpo_lamba_5 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 15000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'State': 'finished',
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.reg_coef': 0.1,
        'config.lamba': 5
    }
    filters_dict_rdpo_lamba_0pt75 = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 15000,
        'config.rdpo_exp_step_size': 0.0001,
        'group': group,
        'config.rdpo_weighted_batches' : False,
        'config.ipo_grad_type': 'linear',
        'config.use_closed_form': True,
        'State': 'finished',
        'config.rdpo_adj': '0',
        'config.deterministic': False,
        'config.reg_coef': 0.1,
        'config.lamba': 0.75
    }
   
   
    # Assume you have the necessary functions and libraries imported
    metrics_to_collect = ['grad_norm', 'train_loss', 'reward_err_1', 'reward_err_2', 'reward_param_1', 'reward_param_2', 'reward_param_3', 'reward_param_4','group_weight_1','group_weight_2','val_loss','train_group_loss_1','train_group_loss_2','val_group_loss_1','val_group_loss_2','hist_group_loss_1','hist_group_loss_2','max_val_grp_loss','max_train_grp_loss','max_reward_err','max_kl_dist']
    # List of filters_dict values
    

    filters_runs_dict={}
    filters_runs_dict['filters_dict_all_runs']=[filters_dict_rdpo_lamba_1, filters_dict_dpo_lamba_1, filters_dict_imp_samp_lamba_1]
    filters_runs_dict['filters_dict_wo_2']=[filters_dict_rdpo_lamba_1_wo_2_extreme, filters_dict_dpo_lamba_1_wo_2_extreme, filters_dict_imp_samp_lamba_1_wo_2_extreme]
    filters_runs_dict['filters_dict_wo_4']=[filters_dict_rdpo_lamba_1_wo_4_extreme, filters_dict_dpo_lamba_1_wo_4_extreme, filters_dict_imp_samp_lamba_1_wo_4_extreme]


    run_filters_name='filters_dict_wo_4'
    unique_part_name='_'.join(run_filters_name.split("_")[2:])
    filters_dicts = filters_runs_dict[run_filters_name]
    


    #filters_dicts = [filters_dict_rdpo_lamba_1,filters_dict_rdpo_lamba_2,filters_dict_rdpo_lamba_5,filters_dict_rdpo_lamba_0]
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
    base_folder = f'wandb-plots/closed_form_lamba/{len(filters_dicts)}_{group}_deterministic_{str(filters_dicts[0]["config.deterministic"])}_{unique_part_name}'
    print(run_filters_name.split('_')[2:],'folder_name')
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
            #values_matrix=prune_bad_runs(values_matrix,0)
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
            n_runs=len(all_metrics_history[metric][i])
            if algo=='rdpo' and filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            if 'd' in algo:
                algo=algo.replace('d','i')
            lamba=filters_dict['config.lamba']
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]
            print(len(avg_values),len(iteration_index),algo)
            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo + '_lamba_' + str(lamba)+'_'+str(n_runs))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo+ '_lamba_' + str(lamba)+'_'+str(n_runs))
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
            n_runs=len(all_metrics_history[metric][i])
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            lamba=filters_dict['config.lamba']
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
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
            n_runs=len(all_metrics_history[metric][i])
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            lamba=filters_dict['config.lamba']
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric+'_'+weight+'_' + algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric+'_'+weight+'_' + algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
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
            n_runs=len(all_metrics_history[metric][i])
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            lamba=filters_dict['config.lamba']
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
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
            n_runs=len(all_metrics_history[metric][i])
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            lamba=filters_dict['config.lamba']
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
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
            n_runs=len(all_metrics_history[metric][i])
            if algo=='rdpo':
                if filters_dict['config.importance_sampling']==True:
                    algo='imp_samp'
                elif filters_dict['config.rdpo_weighted_batches']==True:
                    algo='rdpo_weightbatch'
                else:
                    algo='rdpo_unweightbatch'
            if 'd' in algo:
                algo=algo.replace('d','i')
            lamba=filters_dict['config.lamba']
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Reward Parameters')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/reward_params_3_4')
    plt.close()
    
    colors=['Blues','Oranges']
    plt.figure(figsize=(12, 6))
    for j,metric in enumerate(['group_weight_1', 'group_weight_2']):
        weight=str(weights_array[j])
        color_map=plt.get_cmap(colors[j])
        tot_col=len(filters_dicts)
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            n_runs=len(all_metrics_history[metric][i])
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
            lamba=filters_dict['config.lamba']
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values,color=color_map((i+1)/tot_col), label=metric +'_'+ weight +'_'+ algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, color=color_map((i+1)/tot_col), alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values,color=color_map((i+1)/tot_col), label=metric +'_'+ weight +'_'+ algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values,color=color_map((i+1)/tot_col), alpha=0.2)

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
            n_runs=len(all_metrics_history[metric][i])
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
            lamba=filters_dict['config.lamba']
            if 'ripo' in algo:
                for k in range(mult_runs):
                    rand_index=rand_indices[k]
                    if rand_index>=len(all_metrics_history[metric][i]):
                        rand_index=len(all_metrics_history[metric][i])-1
                    avg_values = all_metrics_history[metric][i][rand_index]
                    if len(avg_values) != len(iteration_index):
                        # Extend avg_values by repeating the last value
                        extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                        plt.plot(iteration_index, extended_avg_values,color=color, label=metric +'_'+ weight +'_'+ algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
                    else:
                        plt.plot(iteration_index, avg_values,color=color, label=metric +'_'+ weight +'_'+ algo+'_lamba_' + str(lamba)+'_'+str(n_runs))

            else:
                avg_values = all_avg_metrics_at_iterations[metric][i]
                sem_values = all_sem_metrics_at_iterations[metric][i]

                if len(avg_values) != len(iteration_index):
                    # Extend avg_values by repeating the last value
                    extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                    extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                    plt.plot(iteration_index, extended_avg_values, label=metric +'_'+ weight +'_'+ algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
                    plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
                else:
                    plt.plot(iteration_index, avg_values, label=metric +'_'+ weight +'_'+ algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
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
            n_runs=len(all_metrics_history[metric][i])
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
            lamba=filters_dict['config.lamba']
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric +'_ratio'+ weight +'_'+ algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric +'_ratio'+ weight +'_'+ algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
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
            n_runs=len(all_metrics_history[metric][i])
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
            lamba=filters_dict['config.lamba']
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
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
            n_runs=len(all_metrics_history[metric][i])
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
            lamba=filters_dict['config.lamba']
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric +'_ratio'+ weight +'_'+ algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric +'_ratio'+ weight +'_'+ algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
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
            n_runs=len(all_metrics_history[metric][i])
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
            lamba=filters_dict['config.lamba']
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric +'_ratio'+ weight +'_'+ algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric +'_ratio'+ weight +'_'+ algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
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
            n_runs=len(all_metrics_history[metric][i])
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
            lamba=filters_dict['config.lamba']
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
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
            n_runs=len(all_metrics_history[metric][i])
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
            lamba=filters_dict['config.lamba']
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo+'_lamba_' + str(lamba)+'_'+str(n_runs))
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
        n_runs=len(all_metrics_history[metric][i])
        if algo=='rdpo':
            if filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            elif filters_dict['config.rdpo_weighted_batches']==True:
                algo='rdpo_weightbatch'
            else:
                algo='rdpo_unweightbatch'
        if 'd' in algo:
            algo=algo.replace('d','i')
        lamba=filters_dict['config.lamba']
        reward_errs_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'reward_err_' in metric]
        reward_errs_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'reward_err_' in metric]
        print(reward_errs_end_avg)
        plt.bar(np.arange(len(reward_errs_end_avg))+i*0.25, height= reward_errs_end_avg,yerr= reward_errs_end_sem,width=0.2,capsize=5,alpha=0.7,label=algo+'_lamba_' + str(lamba)+'_'+str(n_runs))

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
        n_runs=len(all_metrics_history[metric][i])
        if algo=='rdpo':
            if filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            elif filters_dict['config.rdpo_weighted_batches']==True:
                algo='rdpo_weightbatch'
            else:
                algo='rdpo_unweightbatch'
        if 'd' in algo:
            algo=algo.replace('d','i')
        lamba=filters_dict['config.lamba']
        group_loss_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'train_group_loss' in metric]
        group_loss_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'train_group_loss' in metric]
        print(group_loss_end_avg)
        plt.bar(np.arange(len(group_loss_end_avg))+i*0.125, height= group_loss_end_avg,yerr= group_loss_end_sem,width=0.1,capsize=5,alpha=0.7,label=algo+'_lamba_' + str(lamba)+'_'+str(n_runs))

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
        n_runs=len(all_metrics_history[metric][i])
        if algo=='rdpo':
            if filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            elif filters_dict['config.rdpo_weighted_batches']==True:
                algo='rdpo_weightbatch'
            else:
                algo='rdpo_unweightbatch'
        if 'd' in algo:
            algo=algo.replace('d','i')
        lamba=filters_dict['config.lamba']
        group_loss_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'val_group_loss' in metric]
        group_loss_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'val_group_loss' in metric]
        print(group_loss_end_avg)
        plt.bar(np.arange(len(group_loss_end_avg))+i*0.125, height= group_loss_end_avg,yerr= group_loss_end_sem,width=0.1,capsize=5,alpha=0.7,label=algo+'_lamba_' + str(lamba)+'_'+str(n_runs))

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
        n_runs=len(all_metrics_history[metric][i])
        if algo=='rdpo':
            if filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            elif filters_dict['config.rdpo_weighted_batches']==True:
                algo='rdpo_weightbatch'
            else:
                algo='rdpo_unweightbatch'
        if 'd' in algo:
            algo=algo.replace('d','i')
        lamba=filters_dict['config.lamba']
        max_reward_err_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_reward_err' in metric]
        max_reward_err_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_reward_err' in metric]
        print(max_reward_err_end_avg)
        plt.bar(np.arange(len(max_reward_err_end_avg))+i*0.25, height= max_reward_err_end_avg,yerr= max_reward_err_end_sem,width=0.2,capsize=5,alpha=0.7,label=algo+'_lamba_' + str(lamba)+'_'+str(n_runs))

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
        n_runs=len(all_metrics_history[metric][i])
        if algo=='rdpo':
            if filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            elif filters_dict['config.rdpo_weighted_batches']==True:
                algo='rdpo_weightbatch'
            else:
                algo='rdpo_unweightbatch'
        if 'd' in algo:
            algo=algo.replace('d','i')
        lamba=filters_dict['config.lamba']
        max_group_loss_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_train_grp_loss' in metric]
        max_group_loss_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_train_grp_loss' in metric]
        print(max_group_loss_end_avg)
        plt.bar(np.arange(len(max_group_loss_end_avg))+i*0.125, height= max_group_loss_end_avg,yerr= max_group_loss_end_sem,width=0.1,capsize=5,alpha=0.7,label=algo+'_lamba_' + str(lamba)+'_'+str(n_runs))

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
        n_runs=len(all_metrics_history[metric][i])
        if algo=='rdpo':
            if filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            elif filters_dict['config.rdpo_weighted_batches']==True:
                algo='rdpo_weightbatch'
            else:
                algo='rdpo_unweightbatch'
        if 'd' in algo:
            algo=algo.replace('d','i')
        lamba=filters_dict['config.lamba']
        max_group_loss_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_val_grp_loss' in metric]
        max_group_loss_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_val_grp_loss' in metric]
        print(max_group_loss_end_avg)
        plt.bar(np.arange(len(max_group_loss_end_avg))+i*0.125, height= max_group_loss_end_avg,yerr= max_group_loss_end_sem,width=0.1,capsize=5,alpha=0.7,label=algo+'_lamba_' + str(lamba)+'_'+str(n_runs))

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
        n_runs=len(all_metrics_history[metric][i])
        if algo=='rdpo':
            if filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            elif filters_dict['config.rdpo_weighted_batches']==True:
                algo='rdpo_weightbatch'
            else:
                algo='rdpo_unweightbatch'
        if 'd' in algo:
            algo=algo.replace('d','i')
        lamba=filters_dict['config.lamba']
        max_kl_dist_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_kl_dist' in metric]
        max_kl_dist_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_kl_dist' in metric]
        print(max_kl_dist_end_avg)
        plt.bar(np.arange(len(max_kl_dist_end_avg))+i*0.125, height= max_kl_dist_end_avg,yerr= max_kl_dist_end_sem,width=0.1,capsize=5,alpha=0.7,label=algo+'_lamba_' + str(lamba)+'_'+str(n_runs))

        #plt.xticks(np.arange(len(max_group_loss_end_avg))+i*0.125, [f"Group {i+1}" for i in range(len(group_loss_end_avg))])
    plt.title('Max KL distance at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_kl_distance_bars')
    plt.close()