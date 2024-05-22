import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
import ast
import wandb
from algos.linear_bandit.mle import MLERewardLearning
from algos.linear_bandit.pg import PolicyGradient
from algos.linear_bandit.group_dpo import GroupDirectPolicyOptimization
from algos.linear_bandit.group_robust_dpo import GroupRobustDirectPolicyOptimization
from envs.linear_bandit import LinearBandit, ret_feature_func
from envs.group_linear_bandit import GroupLinearBanditSep, GroupLinearBandit, ret_feature_func
from utils.io_utils import save_code, save_config, create_log_dir
from utils.logger import Logger
from utils.collect_data import (
    ret_uniform_policy_group,
    collect_preference_data,
    collect_group_preference_data,
    collect_rl_data,
    merge_datasets,
    pref_to_rl,
)
from collections import defaultdict
from utils.utils import return_apt_weights
import copy 
from utils.utils import softmax
import matplotlib.pyplot as plt
import neatplot
neatplot.set_style()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="group_linear_bandit")
    parser.add_argument("--state_dim", type=int, default=1)
    parser.add_argument("--action_num", type=int, default=4)
    parser.add_argument("--group_num", type=int, default=2)
    parser.add_argument("--agent", type=str, default="pg")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--feature_type", type=str, default='same')
    parser.add_argument("--eval_metric", type=str, default='expectation')
    #parser.add_argument("--flip_feature", action="store_true")

    parser.add_argument("--pref_data_num", type=int, default=500)
    parser.add_argument('--weights',type=str,default='equal')
    parser.add_argument("--val_data_num", type=int, default=50)
    parser.add_argument('--val_weights',type=str,default='equal')
    parser.add_argument("--num_trials_for_eval", type=int, default=1000)
    parser.add_argument('--test_weights',type=str,default='equal')

    parser.add_argument("--mle_num_iters", type=int, default=100)
    parser.add_argument("--mle_adaptive", action="store_true")
    parser.add_argument("--mle_ada_coef", type=float, default=1.0)
    parser.add_argument("--mle_step_size", type=float, default=0.1)

    parser.add_argument("--rl_data_ratio", type=float, default=4)
    parser.add_argument("--reg_coef", type=float, default=1.0)

    parser.add_argument("--dpo_type", type=str,default='dpo')
    parser.add_argument("--dpo_num_iters", type=int, default=200)
    parser.add_argument("--dpo_adaptive", action="store_true")
    parser.add_argument("--dpo_ada_coef", type=float, default=1.0)
    parser.add_argument("--dpo_step_size", type=float, default=0.1)
    parser.add_argument("--rdpo_batch_size", type=int, default=5)
    parser.add_argument("--rdpo_exp_step_size", type=float, default=0.01)
    parser.add_argument("--rdpo_weighted_batches", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--rdpo_adj", type=str, default='0')
    parser.add_argument("--importance_sampling",  type=lambda x: (str(x).lower() == 'true'), default=False)

    parser.add_argument("--pg_num_iters", type=int, default=1000)
    parser.add_argument("--pg_adaptive", action="store_true")
    parser.add_argument("--pg_ada_coef", type=float, default=1.0)
    parser.add_argument("--pg_step_size", type=float, default=0.1)

    parser.add_argument("--wandb_use", action="store_true")
    parser.add_argument("--wandb_key", type=str, default="[key]")
    parser.add_argument("--wandb_entity", type=str, default="robust-rl-project")
    parser.add_argument("--wandb_project", type=str, default="bandits_dpo")
    parser.add_argument("--wandb_group", type=str, default="group1")
    parser.add_argument("--wandb_name", type=str, default="linear_bandits")

    return parser.parse_args()


def get_reward_func(reward_param: np.ndarray, feature_func):
    def reward_func(state, action, group_id):
        feature = feature_func(state, action, group_id)
        rew = np.dot(feature, reward_param)

        return rew

    return reward_func

def set_reward_params(feature_dim: int):
    assert feature_dim in [2, 4, 8, 16]
    if feature_dim == 2:
        rparams = np.array([[5.0, 7.0],[9.0,3.0]], np.float32)
        #rparams = np.array([1.0, 2.0], np.float32)
    elif feature_dim == 4:
        # rparams = np.array([2.0, 1.0, 1.0, 2.0], np.float32)
        rparams = np.array([[1.0, 3.0,1.0, 3.0],[3.0,1.0,3.0,1.0]], np.float32)
        #rparams = np.array([1., 1.2, 0.3, 1.3], np.float32)
    elif feature_dim == 8:
        rparams = np.array([[1.0, 3.0,1.0, 3.0,1.0, 3.0,1.0, 3.0],[3.0,1.0,3.0,1.0,3.0,1.0,3.0,1.0]], np.float32)
        #rparams = np.array([2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0], np.float32)
    elif feature_dim == 16:
        rparams = np.array([[1.0, 3.0,1.0, 3.0,1.0, 3.0,1.0, 3.0,1.0, 3.0,1.0, 3.0,1.0, 3.0,1.0, 3.0],[3.0,1.0,3.0,1.0,3.0,1.0,3.0,1.0,3.0,1.0,3.0,1.0,3.0,1.0,3.0,1.0]], np.float32)
        #rparams = np.array([[1.0, 3.0],[3.0,1.0],[1.0, 3.0],[3.0,1.0],[1.0, 3.0],[3.0,1.0],[1.0, 3.0],[3.0,1.0]], np.float32)
        #rparams = np.array([2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0], np.float32)
    
    assert feature_dim == rparams.shape[1]
    return rparams

def ret_policy(action_num,feature_func,param):
    action_num = action_num
    feature_func = copy.deepcopy(feature_func)
    param = param

    def policy(state: np.ndarray, group_id: int) -> np.ndarray:
        arr = np.zeros(action_num, np.float32)
        for action_idx in range(action_num):
            feature = feature_func(state, action_idx, group_id)
            arr[action_idx] = np.dot(feature, param)
        prob = softmax(arr)

        return prob

    return policy

def plot_opt_actions(test_pref,policy):
    states=[]
    actions=[]
    for transition in test_pref:
        state, action_one, action_two, group_id, pref = (
            transition.state,
            transition.action_0,
            transition.action_1,
            transition.group_id,
            transition.pref,
        )
        pol_action=policy(state,group_id)
        states.append(state)
        actions.append(np.argmax(pol_action))
    plt.figure(figsize=(20, 15))
    #print(states,actions)
    plt.scatter(states,actions,label='action')
    #plt.xticks(x + (bar_width * (num_groups - 1) / 2), weights, fontsize=22)
    plt.xlabel('States', fontsize=36)
    plt.ylabel('Actions', fontsize=36)
    plt.title('Optimal Action', fontsize=38)
    plt.legend(fontsize=34)
    
    neatplot.save_figure(f'actions of optimal policy')

def plot_rewards(test_pref,feature_func,reward_param,action_num,reward_err):
    states=[]
    reward_per_action_0=defaultdict(list)
    reward_per_action_1=defaultdict(list)
    for transition in test_pref:
        state, action_one, action_two, group_id, pref = (
            transition.state,
            transition.action_0,
            transition.action_1,
            transition.group_id,
            transition.pref,
        )
        states.append(state)
        for action in range(action_num):
            #print(feature_func(state,action,group_id),reward_param[group_id])
            reward_per_action_0[action].append(feature_func(state,action,0)@(reward_param[0]))
            reward_per_action_1[action].append(feature_func(state,action,1)@(reward_param[1]))
                                     
    for group in range(2):  # Assuming there are 2 groups
        plt.subplot(2, 1, group + 1)
        for action in range(action_num):
            if group == 0:
                plt.scatter(states, reward_per_action_0[action], label=f'rewards_for_action{action}_group_0')
            else:
                plt.scatter(states, reward_per_action_1[action], label=f'rewards_for_action{action}_group_1')

   

    #plt.suptitle('Rewards for each group', fontsize=20)
    plt.tight_layout()
    #plt.xticks(x + (bar_width * (num_groups - 1) / 2), weights, fontsize=22)
    plt.xlabel('States', fontsize=36)
    plt.ylabel('Rewards', fontsize=36)
    plt.title(f'Rewards for each action', fontsize=38)
    plt.legend(fontsize=34,title=f'Param_[{reward_param[0]},{reward_param[1]}]',title_fontsize=34)
    
    neatplot.save_figure(f'Rewards of each action_{reward_param[0]}_{reward_param[1]}')

def main(args):
    np.random.seed(args.seed)
    #log_dir = create_log_dir(args)
    #save_code(log_dir)
    #save_config(args.__dict__, log_dir)
     
    #print(f"Logging to {log_dir}")
    print("(IB)Seed:"+str(args.seed))
    print("(IB)Data:" + str(args.pref_data_num))
    
    print("args.wandb_use: ", args.wandb_use)
    """
    if args.wandb_use == True:
        print("USING WANDB")
        wandb.login(
            key=args.wandb_key
        )
        if args.dpo_adaptive:
            tags=[args.dpo_num_iters,f"adaptive_{args.dpo_adaptive}",args.ada_coef,args.reg_coef]
        else:
            tags=[f"num_iters_{args.dpo_num_iters}",f"adaptive_{args.dpo_adaptive}",f"step_size_{args.dpo_step_size}",f"beta_{args.reg_coef}"]
        if args.dpo_type=='dpo':
            exp_name=args.wandb_name +"_"+args.dpo_type + "_" + str(args.seed)
        else:
            exp_name=args.wandb_name +"_"+args.dpo_type + "_" + str(args.rdpo_exp_step_size) +"_" + str(args.rdpo_batch_size) + '_' + str(args.rdpo_weighted_batches) + "_" + args.rdpo_adj  + "_" + str(args.seed)
        wandb.init(
            group=f'state_dim{args.state_dim}'+f'action_num{args.action_num}'+f'group_num{args.group_num}'+f'pref_data_num{args.pref_data_num}'+f'weights{args.weights}'+f'feature_type{args.feature_type}'+f'eval_metric{args.eval_metric}',
            entity=args.wandb_entity,
            project=args.wandb_project,
            config=args.__dict__,
            dir=log_dir,
            name=exp_name,
            tags=tags
        )
    """
    state_dim = args.state_dim
    action_num = args.action_num
    group_num = args.group_num

    feature_dim = 2 * args.state_dim
    num_trials_for_eval = args.num_trials_for_eval
    feature_func = ret_feature_func(num_action=action_num, state_dim=state_dim, group_num=group_num,feature_type=args.feature_type)
    # reward_param = np.random.standard_normal(feature_dim)
    # reward_param = np.array([2.0, 1.0, 1.0, 2.0], np.float32)
    reward_param=set_reward_params(feature_dim)
    print(group_num,reward_param)
    assert group_num == np.shape(reward_param)[0], "The feature is invalid."

    # reward_param /= np.sqrt(np.sum(np.square(reward_param)))
    env = GroupLinearBanditSep(
        state_dim,
        action_num,
        group_num,
        reward_param,
        feature_func,
        num_trials_for_eval=num_trials_for_eval,
        eval_metric=args.eval_metric
    )

    weights,val_weights,test_weights=return_apt_weights(args.weights,group_num),return_apt_weights(args.val_weights,group_num),return_apt_weights(args.test_weights,group_num)
    
    opt_policy = env.get_opt_policy()
    
    uniform_policy = ret_uniform_policy_group(action_num)
    #Generate datasets:
    pref_data = collect_group_preference_data(args.pref_data_num, env, weights, uniform_policy)
    val_pref = collect_group_preference_data(args.val_data_num, env, val_weights, uniform_policy)
    test_pref = collect_group_preference_data(args.num_trials_for_eval, env, test_weights, uniform_policy)


    opt_reward = env.evaluate_reward_group_wise(policy=opt_policy,states=test_pref)
    policy_feature_func = ret_feature_func(
        num_action=action_num, state_dim=state_dim, group_num=group_num, feature_type=args.feature_type
    )
    plot_opt_actions(test_pref,opt_policy)
    reward_true_param=env.evaluate_reward_group_wise(policy=ret_policy(action_num,policy_feature_func,reward_param[0]),states=test_pref)
    reward_err=[float((a-b)/a) for a,b in zip(opt_reward,reward_true_param)]
    plot_rewards(test_pref,feature_func,reward_param,action_num,reward_err)
    reward_true_param=env.evaluate_reward_group_wise(policy=ret_policy(action_num,policy_feature_func,np.array([15.0, 75.0], np.float32)),states=test_pref)
    reward_err=[float((a-b)/a) for a,b in zip(opt_reward,reward_true_param)]
    plot_rewards(test_pref,feature_func,np.array([[15.0, 75.0],[15.0,75.0]], np.float32),action_num,reward_err)
    

if __name__ == "__main__":
    main(parse_args())
