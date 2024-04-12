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
    
    filters = {
        # Add additional filters if needed
    }


    #names
    ripo_th='Robust IPO Theory'
    ripo_pr='Robust IPO Prac'
    ipo='IPO'
    imp_samp='Importance_sampling'
    
    setting='uneven_balanced_ipo'
    
    # RIPO swapped
    group_ripo_imbalanced = 'state_dim2action_num8group_num2pref_data_num300weights[0.2,0.8]feature_typeswappedeval_metricargmax_state-1'
    group_ripo_balanced = 'state_dim2action_num8group_num2pref_data_num300weights[0.5,0.5]feature_typeswappedeval_metricargmax_state-1'
    
    # RDPO swapped
    group_rdpo_even_imbalanced = 'state_dim2action_num8group_num2pref_data_num300weights[0.2,0.8]feature_typeswappedeval_metricargmax_iason_even_imbal_osc'
    group_dpo_even_imbalanced = 'state_dim2action_num8group_num2pref_data_num300weights[0.2,0.8]feature_typeswappedeval_metricargmax_iason_even_imbal_osc_dpo'
    group_imp_even_imbalanced = 'state_dim2action_num8group_num2pref_data_num300weights[0.2,0.8]feature_typeswappedeval_metricargmax_iason_even_imbal_osc_imp'

    group_rdpo_uneven_balanced = 'state_dim2action_num8group_num2pref_data_num300weights[0.5,0.5]feature_typeswappedeval_metricargmax_iason_uneven_bal_osc'
    group_dpo_uneven_balanced = 'state_dim2action_num8group_num2pref_data_num300weights[0.5,0.5]feature_typeswappedeval_metricargmax_iason_uneven_bal_osc_dpo'
    group_imp_uneven_balanced = group_dpo_uneven_balanced

    group_rdpo_uneven_imbalanced = 'state_dim2action_num8group_num2pref_data_num300weights[0.2,0.8]feature_typeswappedeval_metricargmax_iason_uneven_imbal_osc'
    group_dpo_uneven_imbalanced = 'state_dim2action_num8group_num2pref_data_num300weights[0.2,0.8]feature_typeswappedeval_metricargmax_iason_uneven_imbal_osc_dpo'
    group_imp_uneven_imbalanced = 'state_dim2action_num8group_num2pref_data_num300weights[0.2,0.8]feature_typeswappedeval_metricargmax_iason_uneven_imbal_osc_imp'

    if 'imbalanced' in setting:
        group_ipo = group_ripo_imbalanced
    else:
        group_ipo = group_ripo_balanced
    
    
    
    weights_array = np.array(group_ipo.split('weights[')[-1].split(']')[0].split(','), dtype=float)
    pref_data_num=group_ipo.split('pref_data_num')[1].split('weights')[0]

    # IPO Filters
    filters_dict_rdpo_lamba_theory_even = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 100,
        'config.rdpo_exp_step_size': 0.01,
        'group': group_ipo,
        'State': 'finished',
        'config.lamba': 0,
        'config.use_uneven_grp': False,
        'config.use_theory': True
    }

    filters_dict_rdpo_lamba_avg_even = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 100,
        'config.rdpo_exp_step_size': 0.01,
        'group': group_ipo,
        'State': 'finished',
        'config.lamba': 0,
        'config.use_uneven_grp': False,
        'config.use_theory': False
    }

    filters_dict_imp_samp_even = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': True,
        'config.dpo_num_iters': 10,
        'group': group_ipo,
        'State': 'finished',
        'config.lamba': 0,
        'config.use_uneven_grp': False,
        'config.importance_sampling_weights': {'$nin': ['0.5,0.5']}
    }

    filters_dict_dpo_even = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': True,
        'config.dpo_num_iters': 10,
        'group': group_ipo,
        'State': 'finished',
        'config.lamba': 0,
        'config.use_uneven_grp': False,
        'config.importance_sampling_weights': {'$in': ['0.5,0.5']}
    }

    filters_dict_rdpo_lamba_theory_uneven = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 100,
        'config.rdpo_exp_step_size': 0.01,
        'group': group_ipo,
        'State': 'finished',
        'config.lamba': 0,
        'config.use_uneven_grp': True,
        'config.use_theory': True
    }

    filters_dict_rdpo_lamba_avg_uneven = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': False,
        'config.dpo_num_iters': 100,
        'config.rdpo_exp_step_size': 0.01,
        'group': group_ipo,
        'State': 'finished',
        'config.lamba': 0,
        'config.use_uneven_grp': True,
        'config.use_theory': False
    }

    filters_dict_imp_samp_uneven = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': True,
        'config.dpo_num_iters': 10,
        'group': group_ipo,
        'State': 'finished',
        'config.lamba': 0,
        'config.use_uneven_grp': True,
        'config.importance_sampling_weights': {'$nin': ['0.5,0.5']}
    }

    filters_dict_dpo_uneven = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': True,
        'config.dpo_num_iters': 10,
        'group': group_ipo,
        'State': 'finished',
        'config.lamba': 0,
        'config.use_uneven_grp': True,
        'config.importance_sampling_weights': {'$in': ['0.5,0.5']}
    }

    # RDPO Filters
    filters_dict_rdpo_even_imbalanced = {'group': group_rdpo_even_imbalanced, 'State': 'finished', 'config.dpo_type': 'rdpo', 'config.importance_sampling': True, 'config.use_theory': False}
    filters_dict_dpo_even_imbalanced = {'group': group_dpo_even_imbalanced, 'State': 'finished', 'config.dpo_type': 'rdpo', 'config.importance_sampling': True, 'config.use_theory': False}
    filters_dict_imp_even_imbalanced = {'group': group_imp_even_imbalanced, 'State': 'finished', 'config.dpo_type': 'rdpo', 'config.importance_sampling': True, 'config.use_theory': False}

    filters_dict_rdpo_uneven_balanced = {'group': group_rdpo_uneven_balanced, 'State': 'finished', 'config.dpo_type': 'rdpo', 'config.importance_sampling': True, 'config.use_theory': False}
    filters_dict_dpo_uneven_balanced = {'group': group_dpo_uneven_balanced, 'State': 'finished', 'config.dpo_type': 'rdpo', 'config.importance_sampling': True, 'config.use_theory': False}

    filters_dict_rdpo_uneven_imbalanced = {'group': group_rdpo_uneven_imbalanced, 'State': 'finished', 'config.dpo_type': 'rdpo', 'config.importance_sampling': True, 'config.use_theory': False}
    filters_dict_dpo_uneven_imbalanced = {'group': group_dpo_uneven_imbalanced, 'State': 'finished', 'config.dpo_type': 'rdpo', 'config.importance_sampling': True, 'config.use_theory': False}
    filters_dict_imp_uneven_imbalanced = {'group': group_imp_uneven_imbalanced, 'State': 'finished', 'config.dpo_type': 'rdpo', 'config.importance_sampling': True, 'config.use_theory': False}

   
    # Assume you have the necessary functions and libraries imported
    metrics_to_collect = ['grad_norm', 'train_loss', 'reward_err_1', 'reward_err_2', 'reward_param_1', 'reward_param_2', 'reward_param_3', 'reward_param_4','group_weight_1','group_weight_2','val_loss','train_group_loss_1','train_group_loss_2','val_group_loss_1','val_group_loss_2','hist_group_loss_1','hist_group_loss_2','max_val_grp_loss','max_train_grp_loss','max_reward_err','max_kl_dist']
    # List of filters_dict values
    filters_dict_even_ipo = [filters_dict_rdpo_lamba_theory_even,filters_dict_rdpo_lamba_avg_even,filters_dict_imp_samp_even,filters_dict_dpo_even]
    filters_dict_uneven_ipo = [filters_dict_rdpo_lamba_theory_uneven,filters_dict_rdpo_lamba_avg_uneven,filters_dict_imp_samp_uneven,filters_dict_dpo_uneven]
    
    filters_dict_even_imbal_rdpo = [filters_dict_rdpo_even_imbalanced,filters_dict_dpo_even_imbalanced,filters_dict_imp_even_imbalanced]
    filters_dict_uneven_bal_rdpo = [filters_dict_rdpo_uneven_balanced,filters_dict_dpo_uneven_balanced]
    filters_dict_uneven_imbal_rdpo = [filters_dict_rdpo_uneven_imbalanced,filters_dict_dpo_uneven_imbalanced,filters_dict_imp_uneven_imbalanced]
    
    if 'ipo' in setting:
        if 'uneven' in setting:
            filters_dicts=filters_dict_uneven_ipo
        else:
            filters_dicts=filters_dict_even_ipo
    elif 'rdpo' in setting:
        if 'even_imbalanced' in setting:
            filters_dicts=filters_dict_even_imbal_rdpo
        elif 'uneven_balanced' in setting:
            filters_dicts=filters_dict_uneven_bal_rdpo
        elif 'uneven_imbalanced' in setting:
            filters_dicts=filters_dict_uneven_imbal_rdpo
    elif 'all' in setting:
        if 'uneven' in setting:
            if 'balanced' in setting:
                filters_dicts=filters_dict_uneven_ipo.extend(filters_dict_uneven_bal_rdpo)
            else:
                filters_dicts=filters_dict_uneven_ipo.extend(filters_dict_uneven_imbal_rdpo)
        else:
            filters_dicts=filters_dict_even_ipo.extend(filters_dict_even_imbal_rdpo)

    # Initialize dictionaries to accumulate metrics data
    all_metrics_history = {metric: [] for metric in metrics_to_collect}
    all_runs=[]
    # Loop through each filters_dict value
    for filters_dict in filters_dicts:
        # Download runs for the current filters_dict
        runs = download_runs(entity, project, filters_dict)
        all_runs.append(runs)
        print(len(runs), type(runs))
        metrics_history = {}

        for metric in metrics_to_collect:
            algo=filters_dict['config.dpo_type']
            data_num=pref_data_num
            if algo=='dpo' and 'group_weight' in metric:
                metrics_history[metric] =[]
                continue
            metrics_history[metric] = process_runs(runs, field=metric, time_field='Iteration')

        # Accumulate metrics data for each configuration
        for metric in metrics_to_collect:
            all_metrics_history[metric].append(metrics_history[metric])
            # all_metrics_history[metric] is a list of dataframes for processed runs
            
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
    base_folder = f'wandb-plots-final/{len(filters_dicts)}_{filters_dicts[0]["group"]}_setting_{setting}'

    # Create the base folder if it doesn't exist
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # Create a subfolder with keys and values
    #subfolder_name = '_'.join([f'{key}-{value}' for key, value in filters_dict[0].items()])
    subfolder_name=filters_dicts[0]['config.dpo_type']+str(len(filters_dict))
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
            #print('Val Matrix: ', values_matrix)
            #print('Type: ', type(values_matrix))
            avg_values = list(map(lambda x: np.mean(x, axis=0).ravel(), values_matrix))
            #print('Avg Values: ', avg_values)
            sem_values = list(map(lambda x: sem(x, axis=0).ravel(), values_matrix))
            all_avg_metrics_at_iterations[metric].append(avg_values)#.ravel())
            all_sem_metrics_at_iterations[metric].append(sem_values)#.ravel())

    
    # Plot metrics at 2000 iterations with error area
    #iteration_index = metrics_history['Iteration'][0]  # Assume iterations are the same for all runs
    print(iteration_index)
    plt.figure(figsize=(12, 6))

    for metric in ['train_loss','val_loss']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            data_num=pref_data_num
            #det_ratio_list=filters_dict['config.deterministic_ratio_list']
            #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
            #lamba=filters_dict['config.lamba']
            if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_th
            elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==False:
                algo=ripo_pr
            elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
                algo=ipo
            else:
                #print(filters_dict['config.importance_sampling_weights'])
                algo=imp_samp
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]
            print(len(avg_values),len(iteration_index),algo)
            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=algo + '_data_num_' + str(data_num) )
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo + '_data_num_' + str(data_num) )
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
            data_num=pref_data_num
            #det_ratio_list=filters_dict['config.deterministic_ratio_list']
            #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
            #lamba=filters_dict['config.lamba']
            if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_th
            elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==False:
                algo=ripo_pr
            elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
                algo=ipo
            else:
                algo=imp_samp
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=metric + algo + '_data_num_' + str(data_num))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo + '_data_num_' + str(data_num) )
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)
    plt.title('Grad Norm')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/grad')
    plt.close()

    #action_type_data=['same','same']
    plt.figure(figsize=(12, 6))
    for j,metric in enumerate(['reward_err_1', 'reward_err_2']):
        weight=str(weights_array[j])
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
            data_num=pref_data_num
            #det_ratio_list=filters_dict['config.deterministic_ratio_list']
            #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
            #lamba=filters_dict['config.lamba']
            if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_th
            elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_pr
            elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
                algo=ipo
            else:
                algo=imp_samp
            #if filters_dict['config.use_uneven_grp']==True:
            #    action_type=action_type_data[j]
            #    weight=weight+'_'+action_type+'_'
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=weight+'_'+ algo + '_data_num_' + str(data_num))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=weight+'_'+ algo + '_data_num_' + str(data_num) )
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
            data_num=pref_data_num
            #det_ratio_list=filters_dict['config.deterministic_ratio_list']
            #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
            #lamba=filters_dict['config.lamba']
            if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_th
            elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_pr
            elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
                algo=ipo
            else:
                algo=imp_samp
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=algo + '_data_num_' + str(data_num))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo + '_data_num_' + str(data_num) )
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
            data_num=pref_data_num
            #det_ratio_list=filters_dict['config.deterministic_ratio_list']
            #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
            #lamba=filters_dict['config.lamba']
            if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_th
            elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_pr
            elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
                algo=ipo
            else:
                algo=imp_samp
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=algo + '_data_num_' + str(data_num))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=algo + '_data_num_' + str(data_num) )
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
            data_num=pref_data_num
            #det_ratio_list=filters_dict['config.deterministic_ratio_list']
            #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
            #lamba=filters_dict['config.lamba']
            if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_th
            elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_pr
            elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
                algo=ipo
            else:
                algo=imp_samp
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=algo + '_data_num_' + str(data_num))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label= algo + '_data_num_' + str(data_num) )
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
            data_num=pref_data_num
            #det_ratio_list=filters_dict['config.deterministic_ratio_list']
            #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
            #lamba=filters_dict['config.lamba']
            #if filters_dict['config.use_uneven_grp']==True:
            #    action_type=action_type_data[j]
            #    weight=weight+'_'+action_type+'_'
            if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_th
            elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_pr
            elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
                algo=ipo
            else:
                algo=imp_samp
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values,color=color, label=weight+'_'+ algo + '_data_num_' + str(data_num))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, color=color, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values,color=color, label=weight+'_'+ algo + '_data_num_' + str(data_num))
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
            data_num=pref_data_num
            #det_ratio_list=filters_dict['config.deterministic_ratio_list']
            #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
            #lamba=filters_dict['config.lamba']
            #if filters_dict['config.use_uneven_grp']==True:
            #    action_type=action_type_data[j]
            #    weight=weight+'_'+action_type+'_'
            if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_th
            elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_pr
            elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
                algo=ipo
            else:
                algo=imp_samp
            if 'robust' in algo:
                for k in range(mult_runs):
                    rand_index=rand_indices[k]
                    avg_values = all_metrics_history[metric][i][rand_index]
                    if len(avg_values) != len(iteration_index):
                        # Extend avg_values by repeating the last value
                        extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                        plt.plot(iteration_index, extended_avg_values,color=color, label=weight+'_'+ algo + '_data_num_' + str(data_num))
                    else:
                        plt.plot(iteration_index, avg_values,color=color, label=weight+'_'+ algo + '_data_num_' + str(data_num))

            else:
                avg_values = all_avg_metrics_at_iterations[metric][i]
                sem_values = all_sem_metrics_at_iterations[metric][i]

                if len(avg_values) != len(iteration_index):
                    # Extend avg_values by repeating the last value
                    extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                    extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                    plt.plot(iteration_index, extended_avg_values, label=weight+'_'+ algo + '_data_num_' + str(data_num) )
                    plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
                else:
                    plt.plot(iteration_index, avg_values, label=weight+'_'+ algo + '_data_num_' + str(data_num) )
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
            data_num=pref_data_num
            #det_ratio_list=filters_dict['config.deterministic_ratio_list']
            #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
            #lamba=filters_dict['config.lamba']
            #if algo=='dpo' and 'group_weight' in metric:
            #    continue
            #if filters_dict['config.use_uneven_grp']==True:
            #    action_type=action_type_data[j]
            #    weight=weight+'_'+action_type+'_'
            if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_th
            elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_pr
            elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
                algo=ipo
            else:
                algo=imp_samp
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=weight +'_'+ algo+ '_data_num_' + str(data_num))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=weight +'_'+ algo+ '_data_num_' + str(data_num))
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
            data_num=pref_data_num
            #det_ratio_list=filters_dict['config.deterministic_ratio_list']
            #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
            #lamba=filters_dict['config.lamba']
            #if algo=='dpo' and 'group_weight' in metric:
            #    continue
            if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_th
            elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_pr
            elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
                algo=ipo
            else:
                algo=imp_samp
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=algo + '_data_num_' + str(data_num) )
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=algo + '_data_num_' + str(data_num) )
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
            data_num=pref_data_num
            #det_ratio_list=filters_dict['config.deterministic_ratio_list']
            #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
            #lamba=filters_dict['config.lamba']
            #if algo=='dpo' and 'group_weight' in metric:
            #    continue
            #if filters_dict['config.use_uneven_grp']==True:
            #    action_type=action_type_data[j]
            #    weight=weight+'_'+action_type+'_'
            if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_th
            elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_pr
            elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
                algo=ipo
            else:
                algo=imp_samp
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=weight +'_'+ algo+ '_data_num_' + str(data_num) )
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=weight +'_'+ algo+ '_data_num_' + str(data_num) )
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
            data_num=pref_data_num
            #det_ratio_list=filters_dict['config.deterministic_ratio_list']
            #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
            #lamba=filters_dict['config.lamba']
            #if filters_dict['config.use_uneven_grp']==True:
            #    action_type=action_type_data[j]
            #    weight=weight+'_'+action_type+'_'
            #if algo=='dpo' and 'group_weight' in metric:
            #    continue
            if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_th
            elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_pr
            elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
                algo=ipo
            else:
                algo=imp_samp
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=weight +'_'+ algo+ '_data_num_' + str(data_num) )
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=weight +'_'+ algo+ '_data_num_' + str(data_num) )
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
            data_num=pref_data_num
            #det_ratio_list=filters_dict['config.deterministic_ratio_list']
            #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
            #lamba=filters_dict['config.lamba']
            #if algo=='dpo' and 'group_weight' in metric:
            #    continue
            if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_th
            elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_pr
            elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
                algo=ipo
            else:
                algo=imp_samp
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=algo + '_data_num_' + str(data_num) )
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=algo + '_data_num_' + str(data_num) )
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
            data_num=pref_data_num
            #det_ratio_list=filters_dict['config.deterministic_ratio_list']
            #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
            #lamba=filters_dict['config.lamba']
            #if algo=='dpo' and 'group_weight' in metric:
            #    continue
            if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_th
            elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
                algo=ripo_pr
            elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
                algo=ipo
            else:
                algo=imp_samp
            avg_values = all_avg_metrics_at_iterations[metric][i]
            sem_values = all_sem_metrics_at_iterations[metric][i]

            if len(avg_values) != len(iteration_index):
                # Extend avg_values by repeating the last value
                extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                plt.plot(iteration_index, extended_avg_values, label=algo + '_data_num_' + str(data_num))
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=algo + '_data_num_' + str(data_num))
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
        data_num=pref_data_num
        #det_ratio_list=filters_dict['config.deterministic_ratio_list']
        #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
        #lamba=filters_dict['config.lamba']
        if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
            algo=ripo_th
        elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
            algo=ripo_pr
        elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
            algo=ipo
        else:
            algo=imp_samp
        reward_errs_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'reward_err_' in metric]
        reward_errs_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'reward_err_' in metric]
        print(reward_errs_end_avg)
        plt.bar(np.arange(len(reward_errs_end_avg))+i*0.25, height=[i.item() for i in reward_errs_end_avg],yerr=[i.item() for i in reward_errs_end_sem],width=0.2,capsize=5,alpha=0.7,label=algo+ '_data_num_' + str(data_num) )

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
        data_num=pref_data_num
        #det_ratio_list=filters_dict['config.deterministic_ratio_list']
        #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
        #lamba=filters_dict['config.lamba']
        if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
            algo=ripo_th
        elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
            algo=ripo_pr
        elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
            algo=ipo
        else:
            algo=imp_samp
        group_loss_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'train_group_loss' in metric]
        group_loss_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'train_group_loss' in metric]
        print(group_loss_end_avg)
        plt.bar(np.arange(len(group_loss_end_avg))+i*0.125, height=[i.item() for i in group_loss_end_avg],yerr=[i.item() for i in group_loss_end_sem],width=0.1,capsize=5,alpha=0.7,label=algo+ '_data_num_' + str(data_num))

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
        data_num=pref_data_num
        #det_ratio_list=filters_dict['config.deterministic_ratio_list']
        #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
        #lamba=filters_dict['config.lamba']
        if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
            algo=ripo_th
        elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
            algo=ripo_pr
        elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
            algo=ipo
        else:
            algo=imp_samp
        group_loss_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'val_group_loss' in metric]
        group_loss_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'val_group_loss' in metric]
        print(group_loss_end_avg)
        plt.bar(np.arange(len(group_loss_end_avg))+i*0.125, height=[i.item() for i in group_loss_end_avg],yerr=[i.item() for i in group_loss_end_sem],width=0.1,capsize=5,alpha=0.7,label=algo+ '_data_num_' + str(data_num) )

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
        data_num=pref_data_num
        #det_ratio_list=filters_dict['config.deterministic_ratio_list']
        #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
        #lamba=filters_dict['config.lamba']
        if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
            algo=ripo_th
        elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
            algo=ripo_pr
        elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
            algo=ipo
        else:
            algo=imp_samp
        max_reward_err_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_reward_err' in metric]
        max_reward_err_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_reward_err' in metric]
        print(max_reward_err_end_avg)
        plt.bar(np.arange(len(max_reward_err_end_avg))+i*0.25, height=[i.item() for i in max_reward_err_end_avg],yerr=[i.item() for i in max_reward_err_end_sem],width=0.2,capsize=5,alpha=0.7,label=algo+ '_data_num_' + str(data_num) )

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
        data_num=pref_data_num
        #det_ratio_list=filters_dict['config.deterministic_ratio_list']
        #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
        #lamba=filters_dict['config.lamba']
        if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
            algo=ripo_th
        elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
            algo=ripo_pr
        elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
            algo=ipo
        else:
            algo=imp_samp
        max_group_loss_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_train_grp_loss' in metric]
        max_group_loss_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_train_grp_loss' in metric]
        print(max_group_loss_end_avg)
        plt.bar(np.arange(len(max_group_loss_end_avg))+i*0.125, height=[i.item() for i in max_group_loss_end_avg],yerr=[i.item() for i in max_group_loss_end_sem],width=0.1,capsize=5,alpha=0.7,label=algo+ '_data_num_' + str(data_num) )

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
        data_num=pref_data_num
        #det_ratio_list=filters_dict['config.deterministic_ratio_list']
        #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
        #lamba=filters_dict['config.lamba']
        if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
            algo=ripo_th
        elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
            algo=ripo_pr
        elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
            algo=ipo
        else:
            algo=imp_samp
        max_group_loss_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_val_grp_loss' in metric]
        max_group_loss_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_val_grp_loss' in metric]
        print(max_group_loss_end_avg)
        plt.bar(np.arange(len(max_group_loss_end_avg))+i*0.125, height=[i.item() for i in max_group_loss_end_avg],yerr=[i.item() for i in max_group_loss_end_sem],width=0.1,capsize=5,alpha=0.7,label=algo+ '_data_num_' + str(data_num) )

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
        data_num=pref_data_num
        #det_ratio_list=filters_dict['config.deterministic_ratio_list']
        #val_det_ratio_list=filters_dict['config.val_deterministic_ratio_list']
        #lamba=filters_dict['config.lamba']
        if filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
            algo=ripo_th
        elif filters_dict['config.importance_sampling']==False and filters_dict['config.use_theory']==True:
            algo=ripo_pr
        elif filters_dict['config.importance_sampling']==True and filters_dict['config.importance_sampling_weights']=={'$in': ['0.5,0.5']}:
            algo=ipo
        else:
            algo=imp_samp
        max_kl_dist_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_kl_dist' in metric]
        max_kl_dist_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metrics_to_collect if 'max_kl_dist' in metric]
        print(max_kl_dist_end_avg)
        plt.bar(np.arange(len(max_kl_dist_end_avg))+i*0.125, height=[i.item() for i in max_kl_dist_end_avg],yerr=[i.item() for i in max_kl_dist_end_sem],width=0.1,capsize=5,alpha=0.7,label=algo+ '_data_num_' + str(data_num) )

        #plt.xticks(np.arange(len(max_group_loss_end_avg))+i*0.125, [f"Group {i+1}" for i in range(len(group_loss_end_avg))])
    plt.title('Max KL distance at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_kl_distance_bars')
    plt.close()
