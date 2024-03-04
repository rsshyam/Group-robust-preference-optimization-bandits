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

"""
def collect_data(entity, project, group, filters):
    # Log in to W&B (you may need to authenticate with your API key)
    wandb.login()

    # Define the filters
    filters_dict = {
        'config.dpo_type': 'rdpo',
        'config.importance_sampling': True,
        'config.dpo_num_iters': 20000,
        'group': group
    }
    filters_dict.update(filters)

    # Initialize the W&B API with entity, project, and group
    #api = wandb.Api(entity=entity, project=project)

    # Query runs based on filters
    #runs = api.runs(group=group, filters=filters_dict)
    api = wandb.Api()
    #entity, project = "<entity>", "<project>"
    runs = api.runs(path=f"{entity}/{project}",filters=filters_dict)
    #print(runs)
    #runs=runs(group=group,filter=filters_dict)
    #runs=(entity=entity, project=project, filters=filters_dict)
    # Collect and print data
    data = []
    count=0
    for run in runs:
        if run.group==group:
            count+=1
            data.append({
                'run_id': run.id,
                'config': run.config,
                'summary': run.summary,
                'history':run.history
            })
    print(count)
    return data
"""
if __name__ == "__main__":
    # Specify the parameters for your data collection
    entity = 'robust-rl-project'
    project = 'bandits_dpo'
    group = 'state_dim2action_num8group_num2pref_data_num60weights[0.2,0.8]feature_typeflippedeval_metricargmax'
    # Generate a list of strings with weights ranging from 0.1 to 0.9 and summing to 1
    groups_list = []
    weights_list = []
    for i in range(1, 3):
        weight_pair = f'[0.{i},0.{10-i}]'
        modified_group = group.replace('[0.2,0.8]', weight_pair)
        groups_list.append(modified_group)
        weights_list.append(weight_pair)
    print(groups_list)
    all_group_runs = {group: [] for group in groups_list}
    all_group_metrics_history = {}
    all_group_avg_metrics_at_iterations = {}
    all_group_sem_metrics_at_iterations = {}
    for group in groups_list:
        weights_array = np.array(group.split('weights[')[-1].split(']')[0].split(','), dtype=float)
        filters = {
            # Add additional filters if needed
        }

        # Define the filters
        filters_dict_rdpo = {
            'config.dpo_type': 'rdpo',
            'config.importance_sampling': False,
            'config.dpo_num_iters': 15000,
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
            'config.dpo_num_iters': 15000,
            'config.rdpo_exp_step_size': 0.0001,
            'group': group,
            'State' : 'finished',
            'config.rdpo_weighted_batches' : False,
            'config.ipo_grad_type': 'linear',
            'config.use_closed_form': True,
            'config.rdpo_adj': '0',
            'config.deterministic': True
        }
        filters_dict_dpo = {
            'config.dpo_type': 'dpo',
            'config.dpo_num_iters': 15000,
            'config.ipo_grad_type': 'Regression',
            'group': group,
            'State': 'finished',
            'config.deterministic': True
        }
        filters_dict_imp_samp = {
            'config.dpo_type': 'rdpo',
            'config.importance_sampling': True,
            'config.dpo_num_iters': 15000,#15000 for pref data 120
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
        filters_dicts = [filters_dict_rdpo_unweighted,filters_dict_dpo,filters_dict_imp_samp]
        # Initialize dictionaries to accumulate metrics data
        all_metrics_history = {metric: [] for metric in metrics_to_collect}
        # Loop through each filters_dict value
        for filters_dict in filters_dicts:
            # Download runs for the current filters_dict
            runs = download_runs(entity, project, filters_dict)
            all_group_runs[group].append(runs)
            print(len(all_group_runs[group]))
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
        all_group_metrics_history[group]=all_metrics_history


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
        all_group_avg_metrics_at_iterations[group] = all_avg_metrics_at_iterations
        all_group_sem_metrics_at_iterations[group] = all_sem_metrics_at_iterations
    iteration_len=0
    iteration_index=0
    for runs in all_group_runs[group]:
        for run in runs:
            iteration_index_1=run[['Iteration']].dropna().values.ravel()
            print(iteration_index_1)
            if len(iteration_index_1)>iteration_len:
                iteration_len=len(iteration_index_1)
                iteration_index=iteration_index_1
 
    #print(metrics_history)



    # Specify the base folder path
    base_folder = f'wandb-plots/closed_form_multi/{len(filters_dicts)}_{len(groups_list)}'

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






    '''
    # Collect data
    collected_data = collect_data(entity, project, group, filters)
    count=0
    # Print the collected data
    for entry in collected_data:
        count+=1
        print(f"Run ID: {entry['run_id']}")
        print(f"Config: {entry['config']}")
        print(f"Summary: {entry['summary']}")
        print(f"History: {entry['history']().keys()}")
        print("\n" + "="*50 + "\n")
        for key in entry['history']().keys():
            print(entry['history']()[key])
        #print(entry['history']()['reward_err_1'])
    print(count)
    # Extract and average metrics
    metrics_to_collect = ['grad_norm', 'train_loss', 'reward_err_1', 'reward_err_2', 'reward_param_1', 'reward_param_2', 'reward_param_3', 'reward_param_4']

    avg_metrics = {metric: [] for metric in metrics_to_collect}

    metrics_at_iterations = {metric: [] for metric in metrics_to_collect}

    hist_at_iterations = {metric: [] for metric in metrics_to_collect}

    for metric in metrics_to_collect:
        hist_value=process_runs(collected_data,metric,time_field='iteration')
        if hist_value is not None:
            metrics_at_iterations[metric].append(values)

    for entry in collected_data:
        for metric in metrics_to_collect:
            hist_value=process_runs()
            values = entry['summary'].get(metric)
            if values is not None:
                metrics_at_iterations[metric].append(values)
    '''
     # Plot all configurations for each metric in the same graph
    # Calculate average and standard error
   

    
    # Plot metrics at 2000 iterations with error area
    #iteration_index = metrics_history['Iteration'][0]  # Assume iterations are the same for all runs
    print(iteration_index)



    plt.figure(figsize=(12, 6))
    # Bar plot for reward_errs at the end
    for i, filters_dict in enumerate(filters_dicts):
        reward_errs_end_avg_grp=[]
        reward_errs_end_sem_grp=[]
        algo=filters_dict['config.dpo_type']
        if algo=='rdpo':
            if filters_dict['config.importance_sampling']==True:
                algo='imp_samp'
            elif filters_dict['config.rdpo_weighted_batches']==True:
                algo='rdpo_weightbatch'
            else:
                algo='rdpo_unweightbatch'
        if 'd' in algo:
            algo=algo.replace('d','i')
        for z,group in enumerate(groups_list):
            reward_errs_end_avg = [all_group_avg_metrics_at_iterations[group][metric][i][-1] for metric in metrics_to_collect if 'max_reward_err' in metric]
            reward_errs_end_sem = [all_group_sem_metrics_at_iterations[group][metric][i][-1] for metric in metrics_to_collect if 'max_reward_err' in metric]
            print(reward_errs_end_avg)
            reward_errs_end_avg_grp.append(reward_errs_end_avg[0])
            reward_errs_end_sem_grp.append(reward_errs_end_sem[0])
        print(reward_errs_end_avg_grp)
        plt.bar(np.arange(len(reward_errs_end_avg_grp))+i*0.15, height= reward_errs_end_avg_grp,yerr= reward_errs_end_sem_grp,width=0.05,capsize=5,alpha=0.7,label=algo)
            
        plt.xticks(np.arange(len(reward_errs_end_avg_grp))+i*0.15, [f"ratio_{weights_array[i]}" for i in range(len(reward_errs_end_avg_grp))])
    plt.title('Reward Errors at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/reward_bars')









    """
    plt.figure(figsize=(12, 6))

    for metric in ['train_loss','val_loss']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
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
                plt.plot(iteration_index, extended_avg_values, label=metric + algo)
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo)
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Train and Val Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/train_val')

    plt.figure(figsize=(12, 6))

    for metric in ['grad_norm']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
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
                plt.plot(iteration_index, extended_avg_values, label=metric + algo)
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo)
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)
    plt.title('Grad Norm')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/grad')
    
    plt.figure(figsize=(12, 6))
    for j,metric in enumerate(['reward_err_1', 'reward_err_2']):
        weight=str(weights_array[j])
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
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
                plt.plot(iteration_index, extended_avg_values, label=metric+'_'+weight+'_' + algo)
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric+'_'+weight+'_' + algo)
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Group wise Reward Errors')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/reward_err')

    plt.figure(figsize=(12, 6))
    for metric in ['max_reward_err']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
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
                plt.plot(iteration_index, extended_avg_values, label=metric + algo)
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo)
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Max Reward Errors')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_reward_err')

    plt.figure(figsize=(12, 6))
    for metric in ['reward_param_1', 'reward_param_2']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
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
                plt.plot(iteration_index, extended_avg_values, label=metric + algo)
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo)
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)
    plt.title('Reward Parameters')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/reward_params_1_2')

    plt.figure(figsize=(12, 6))
    for metric in ['reward_param_3', 'reward_param_4']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
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
                plt.plot(iteration_index, extended_avg_values, label=metric + algo)
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo)
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Reward Parameters')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/reward_params_3_4')
    
    colors=['blue','orange']
    plt.figure(figsize=(12, 6))
    for j,metric in enumerate(['group_weight_1', 'group_weight_2']):
        weight=str(weights_array[j])
        color=colors[j]
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
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
                plt.plot(iteration_index, extended_avg_values,color=color, label=metric +'_'+ weight +'_'+ algo)
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, color=color, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values,color=color, label=metric +'_'+ weight +'_'+ algo)
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values,color=color, alpha=0.2)

    plt.title('Group Weights')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/group_weights')

    mult_runs=5
    colors=['blue','orange']
    rand_indices=np.random.choice(20,size=mult_runs)
    plt.figure(figsize=(12, 6))
    for j,metric in enumerate(['group_weight_1', 'group_weight_2']):
        weight=str(weights_array[j])
        color=colors[j]
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
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
                        plt.plot(iteration_index, extended_avg_values,color=color, label=metric +'_'+ weight +'_'+ algo)
                    else:
                        plt.plot(iteration_index, avg_values,color=color, label=metric +'_'+ weight +'_'+ algo)

            else:
                avg_values = all_avg_metrics_at_iterations[metric][i]
                sem_values = all_sem_metrics_at_iterations[metric][i]

                if len(avg_values) != len(iteration_index):
                    # Extend avg_values by repeating the last value
                    extended_avg_values = np.append(avg_values, [avg_values[-1]] * (len(iteration_index) - len(avg_values)))
                    extended_sem_values = np.append(sem_values, [sem_values[-1]] * (len(iteration_index) - len(sem_values)))
                    plt.plot(iteration_index, extended_avg_values, label=metric +'_'+ weight +'_'+ algo)
                    plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
                else:
                    plt.plot(iteration_index, avg_values, label=metric +'_'+ weight +'_'+ algo)
                    plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Group Weights Multiple Runs')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/group_weights_multiple_runs')

    plt.figure(figsize=(12, 6))
    for j,metric in enumerate(['train_group_loss_1','train_group_loss_2']):
        weight=str(weights_array[j])
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
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
                plt.plot(iteration_index, extended_avg_values, label=metric +'_ratio'+ weight +'_'+ algo)
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric +'_ratio'+ weight +'_'+ algo)
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Group Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/group_train_loss')


    plt.figure(figsize=(12, 6))
    for metric in ['max_train_grp_loss']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
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
                plt.plot(iteration_index, extended_avg_values, label=metric + algo)
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo)
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)
    plt.title('Max train Group Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_group_train_loss')

    plt.figure(figsize=(12, 6))
    for j,metric in enumerate(['hist_group_loss_1','hist_group_loss_2']):
        weight=str(weights_array[j])
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
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
                plt.plot(iteration_index, extended_avg_values, label=metric +'_ratio'+ weight +'_'+ algo)
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric +'_ratio'+ weight +'_'+ algo)
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Group Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/hist_group_loss')

    plt.figure(figsize=(12, 6))
    for j,metric in enumerate(['val_group_loss_1','val_group_loss_2']):
        weight=str(weights_array[j])
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
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
                plt.plot(iteration_index, extended_avg_values, label=metric +'_ratio'+ weight +'_'+ algo)
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric +'_ratio'+ weight +'_'+ algo)
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Group Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/group_val_loss')


    plt.figure(figsize=(12, 6))
    for metric in ['max_val_grp_loss']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
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
                plt.plot(iteration_index, extended_avg_values, label=metric + algo)
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo)
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)

    plt.title('Max Val Group Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_group_val_loss')


    plt.figure(figsize=(12, 6))
    for metric in ['max_kl_dist']:
        for i, filters_dict in enumerate(filters_dicts):
            algo=filters_dict['config.dpo_type']
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
                plt.plot(iteration_index, extended_avg_values, label=metric + algo)
                plt.fill_between(iteration_index, extended_avg_values - extended_sem_values, extended_avg_values + extended_sem_values, alpha=0.2)
            else:
                plt.plot(iteration_index, avg_values, label=metric + algo)
                plt.fill_between(iteration_index, avg_values - sem_values, avg_values + sem_values, alpha=0.2)


    plt.title('Max KL Distance')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_kl_dist')


    
    plt.figure(figsize=(12, 6))
    # Bar plot for reward_errs at the end
    for i, filters_dict in enumerate(filters_dicts):
        algo=filters_dict['config.dpo_type']
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
        plt.bar(np.arange(len(reward_errs_end_avg))+i*0.25, height= reward_errs_end_avg,yerr= reward_errs_end_sem,width=0.2,capsize=5,alpha=0.7,label=algo)

        plt.xticks(np.arange(len(reward_errs_end_avg))+i*0.25, [f"Group {i+1}_ratio_{weights_array[i]}" for i in range(len(reward_errs_end_avg))])
    plt.title('Reward Errors at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/reward_bars')

    plt.figure(figsize=(12, 6))
    # Bar plot for train_group_loss at the end
    for i, filters_dict in enumerate(filters_dicts):
        algo=filters_dict['config.dpo_type']
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
        plt.bar(np.arange(len(group_loss_end_avg))+i*0.125, height= group_loss_end_avg,yerr= group_loss_end_sem,width=0.1,capsize=5,alpha=0.7,label=algo)

        plt.xticks(np.arange(len(group_loss_end_avg))+i*0.125, [f"Group {i+1}_ratio_{weights_array[i]}" for i in range(len(group_loss_end_avg))])
    plt.title('Group Train Loss at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/train_group_loss_bars')

    plt.figure(figsize=(12, 6))
    # Bar plot for val_group_loss at the end
    for i, filters_dict in enumerate(filters_dicts):
        algo=filters_dict['config.dpo_type']
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
        plt.bar(np.arange(len(group_loss_end_avg))+i*0.125, height= group_loss_end_avg,yerr= group_loss_end_sem,width=0.1,capsize=5,alpha=0.7,label=algo)

        plt.xticks(np.arange(len(group_loss_end_avg))+i*0.125, [f"Group {i+1}_ratio_{weights_array[i]}" for i in range(len(group_loss_end_avg))])
    plt.title('Group Validation Loss at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/val_group_loss_bars')
     
    #'max_val_grp_loss','max_train_grp_loss','max_reward_err','max_kl_dist'
    plt.figure(figsize=(12, 6))
    # Bar plot for reward_errs at the end
    for i, filters_dict in enumerate(filters_dicts):
        algo=filters_dict['config.dpo_type']
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
        plt.bar(np.arange(len(max_reward_err_end_avg))+i*0.25, height= max_reward_err_end_avg,yerr= max_reward_err_end_sem,width=0.2,capsize=5,alpha=0.7,label=algo)

        #plt.xticks(np.arange(len(max_reward_err_end_avg))+i*0.25, [f"Group {i+1}" for i in range(len(max_reward_err_end_avg))])
    plt.title('Max Reward Error at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_reward_bars')

    plt.figure(figsize=(12, 6))
    # Bar plot for train_group_loss at the end
    for i, filters_dict in enumerate(filters_dicts):
        algo=filters_dict['config.dpo_type']
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
        plt.bar(np.arange(len(max_group_loss_end_avg))+i*0.125, height= max_group_loss_end_avg,yerr= max_group_loss_end_sem,width=0.1,capsize=5,alpha=0.7,label=algo)

        #plt.xticks(np.arange(len(group_loss_end_avg))+i*0.125, [f"Group {i+1}" for i in range(len(group_loss_end_avg))])
    plt.title('Max Group Train Loss at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_train_group_loss_bars')

    plt.figure(figsize=(12, 6))
    # Bar plot for val_group_loss at the end
    for i, filters_dict in enumerate(filters_dicts):
        algo=filters_dict['config.dpo_type']
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
        plt.bar(np.arange(len(max_group_loss_end_avg))+i*0.125, height= max_group_loss_end_avg,yerr= max_group_loss_end_sem,width=0.1,capsize=5,alpha=0.7,label=algo)

        #plt.xticks(np.arange(len(max_group_loss_end_avg))+i*0.125, [f"Group {i+1}" for i in range(len(group_loss_end_avg))])
    plt.title('Max Group Validation Loss at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_val_group_loss_bars')

    plt.figure(figsize=(12, 6))
    # Bar plot for val_group_loss at the end
    for i, filters_dict in enumerate(filters_dicts):
        algo=filters_dict['config.dpo_type']
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
        plt.bar(np.arange(len(max_kl_dist_end_avg))+i*0.125, height= max_kl_dist_end_avg,yerr= max_kl_dist_end_sem,width=0.1,capsize=5,alpha=0.7,label=algo)

        #plt.xticks(np.arange(len(max_group_loss_end_avg))+i*0.125, [f"Group {i+1}" for i in range(len(group_loss_end_avg))])
    plt.title('Max KL distance at the End')
    plt.ylabel('Value')
    plt.legend()
    neatplot.save_figure(f'{subfolder_path}/max_kl_distance_bars')
    """
