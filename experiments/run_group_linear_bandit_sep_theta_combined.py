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
#from envs.linear_bandit import LinearBandit, ret_feature_func
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
    collect_group_preference_data_partial_deterministic,
    collect_group_preference_data_wth_deterministic_list
)
from utils.utils import return_apt_weights
import copy 
from utils.utils import softmax




def str_to_bool_list(s):
    """Convert a string representation of a list to a list of boolean values."""
    try:
        bool_list = ast.literal_eval(s)
        if isinstance(bool_list, list):
            return [bool(item) for item in bool_list]
        else:
            raise ValueError("Input is not a list.")
    except (SyntaxError, ValueError) as e:
        raise ValueError("Invalid boolean list: {}".format(s))
    

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
    parser.add_argument("--eval_metric_prob", type=str, default='KL')
    parser.add_argument("--val_deterministic", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--deterministic_ratio", type=float, default=0)
    parser.add_argument('--deterministic_list', nargs='+', type=str_to_bool_list, default='[False, False]', help="List of true/false values")
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
    parser.add_argument("--importance_sampling_weights",  type=str, default='None')
    parser.add_argument("--ipo_grad_type",  type=str, default='justdpo')
    parser.add_argument("--param_limit",type=int,default=1)
    parser.add_argument("--use_closed_form",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--lamba",type=float,default=0)

    parser.add_argument("--pg_num_iters", type=int, default=1000)
    parser.add_argument("--pg_adaptive", action="store_true")
    parser.add_argument("--pg_ada_coef", type=float, default=1.0)
    parser.add_argument("--pg_step_size", type=float, default=0.1)

    parser.add_argument("--wandb_use", action="store_true")
    parser.add_argument("--wandb_key", type=str, default="eb687170e674596d211e8f521a3524aac14a07db")
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
        rparams = np.array([[1.0, 2.0],[2.0,1.0]], np.float32)
        #rparams = np.array([1.0, 2.0], np.float32)
    elif feature_dim == 4:
        # rparams = np.array([2.0, 1.0, 1.0, 2.0], np.float32)
        rparams = np.array([[1,3,1, 3],[3,1,3,1]], np.float32)
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

def main(args):
    np.random.seed(args.seed)
    log_dir = create_log_dir(args)
    save_code(log_dir)
    save_config(args.__dict__, log_dir)
     
    print(f"Logging to {log_dir}")
    print("(IB)Seed:"+str(args.seed))
    print("(IB)Data:" + str(args.pref_data_num))
    
    print("args.wandb_use: ", args.wandb_use)
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

    logger = Logger(log_dir)
    writer = SummaryWriter(log_dir)

    state_dim = args.state_dim
    action_num = args.action_num
    group_num = args.group_num

    feature_dim = 2 * args.state_dim
    num_trials_for_eval = args.num_trials_for_eval
    feature_func = ret_feature_func(num_action=action_num, state_dim=state_dim, group_num=group_num,feature_type=args.feature_type)
    # reward_param = np.random.standard_normal(feature_dim)
    # reward_param = np.array([2.0, 1.0, 1.0, 2.0], np.float32)
    reward_param=set_reward_params(feature_dim)
    wandb.config['true_reward_params']=reward_param
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
        eval_metric=args.eval_metric,
        eval_metric_prob=args.eval_metric_prob
    )

    weights,val_weights,test_weights=return_apt_weights(args.weights,group_num),return_apt_weights(args.val_weights,group_num),return_apt_weights(args.test_weights,group_num)
    
    opt_policy = env.get_opt_policy()
    
    uniform_policy = ret_uniform_policy_group(action_num)
    #Generate datasets:
    #print(args.deterministic_list)
    pref_data = collect_group_preference_data(args.pref_data_num, env, weights, uniform_policy,deterministic=False)
    val_pref = collect_group_preference_data(args.val_data_num, env, val_weights, uniform_policy,deterministic=args.val_deterministic)
    test_pref = collect_group_preference_data(args.num_trials_for_eval, env, test_weights, uniform_policy,deterministic=True)
    val_pref = test_pref
    opt_reward = env.evaluate_reward_group_wise(policy=opt_policy,states=test_pref)

    unif_policy_rew = env.evaluate_reward_group_wise(policy=uniform_policy,states=test_pref)

    formatted_opt_reward = ", ".join([f"{reward:.4f}" for reward in opt_reward])
    formatted_unif_policy_rew = ", ".join([f"{reward:.4f}" for reward in unif_policy_rew])
    logger.info(
        f"optimal policy reward: {formatted_opt_reward}, uniform policy reward: {unif_policy_rew}."
    )

    # learn the reward function
    reward_model = MLERewardLearning(
        feature_func,
        feature_dim,
        args.mle_step_size,
        args.mle_num_iters,
        args.mle_adaptive,
        args.mle_ada_coef,
    )
    loss, l2_dist, acc = reward_model.train_by_cvxpy_group(
        dataset=pref_data, true_reward_param=reward_param
    )
    logger.info(f"Reward loss: {loss:.4f}, l2 distance: {l2_dist:.4f}, acc: {acc:.2f}.")

    learned_reward_func = reward_model.get_reward_func
    learned_reward_param = reward_model.get_reward_param
    logger.info("True reward parameter: {}".format(reward_param))
    logger.info("Learned reward parameter: {}".format(learned_reward_param))

    # Oracle test
    learned_env = GroupLinearBandit(
        state_dim,
        action_num,
        group_num,
        learned_reward_param,
        feature_func,
        num_trials_for_eval=num_trials_for_eval,
    )
    learned_oracle_opt_policy = learned_env.get_opt_policy()
    learned_oracle_opt_reward = env.evaluate_reward_group_wise(policy=learned_oracle_opt_policy,states=test_pref)

    formatted_learned_oracle_opt_reward  = ", ".join([f"{reward:.4f}" for reward in learned_oracle_opt_reward])
    logger.info(f"Learned oracle reward: {formatted_learned_oracle_opt_reward}")

    # Train the RL on the preference data
    logger.info(f"Train a policy solely on the preference data (DPO).")
    # learn the policy
    policy_feature_func = ret_feature_func(
        num_action=action_num, state_dim=state_dim, group_num=group_num, feature_type=args.feature_type
    )
    if args.dpo_type == 'dpo':
        agent = GroupDirectPolicyOptimization(
            state_dim=state_dim,
            action_num=action_num,
            group_num=group_num,
            feature_dim=feature_dim,
            feature_func=policy_feature_func,
            ref_policy=uniform_policy,
            reg_coef=args.reg_coef,
            step_size=args.dpo_step_size,
            num_iters=args.dpo_num_iters,
            is_adaptive=args.dpo_adaptive,
            ada_coef=args.dpo_ada_coef,
            logger=logger,
            wandb_use=args.wandb_use,
            ipo_grad_type=args.ipo_grad_type,
            param_limit=args.param_limit,
            lamba=args.lamba
        )
    elif args.dpo_type == 'rdpo':
        agent =  GroupRobustDirectPolicyOptimization(
            state_dim=state_dim,
            action_num=action_num,
            group_num=group_num,
            feature_dim=feature_dim,
            feature_func=policy_feature_func,
            ref_policy=uniform_policy,
            reg_coef=args.reg_coef,
            step_size=args.dpo_step_size,
            num_iters=args.dpo_num_iters,
            is_adaptive=args.dpo_adaptive,
            ada_coef=args.dpo_ada_coef,
            batch_size=args.rdpo_batch_size,
            exp_step_size=args.rdpo_exp_step_size,
            logger=logger,
            wandb_use=args.wandb_use,
            weighted_batches=args.rdpo_weighted_batches,
            adj=args.rdpo_adj,
            importance_sampling=args.importance_sampling,
            importance_sampling_weights=args.importance_sampling_weights,
            ipo_grad_type=args.ipo_grad_type,
            param_limit=args.param_limit,
            use_closed_form=args.use_closed_form,
            lamba=args.lamba
        )
    else:
        agent = GroupDirectPolicyOptimization(
            state_dim=state_dim,
            action_num=action_num,
            group_num=group_num,
            feature_dim=feature_dim,
            feature_func=policy_feature_func,
            ref_policy=uniform_policy,
            reg_coef=args.reg_coef,
            step_size=args.dpo_step_size,
            num_iters=args.dpo_num_iters,
            is_adaptive=args.dpo_adaptive,
            ada_coef=args.dpo_ada_coef,
            logger=logger,
            wandb_use=args.wandb_use,
            ipo_grad_type=args.ipo_grad_type,
            param_limit=args.param_limit,
            lamba=args.lamba,
            train_agent=False
        )

    # reward = agent.train_by_cvxpy(dataset=pref_data, env=env)
    if agent.train_agent==True:
        reward = agent.train(dataset=pref_data, val_dataset=val_pref,test_dataset=test_pref, env=env, optimal_reward=opt_reward)
    else:
        reward = agent.random_train(dataset=pref_data, val_dataset=val_pref,test_dataset=test_pref, env=env, optimal_reward=opt_reward)
    formatted_reward  = ", ".join([f"{reward:.4f}" for reward in reward])
    rew_error = [float((a-b)/a) for a,b in zip(opt_reward,reward)]
    formatted_rew_error  = ", ".join([f"{reward:.4f}" for reward in rew_error])
    policy_param = agent.get_param
    logger.info(
        f"Policy parameter learned solely on the preference data {args.dpo_type}: {policy_param}."
    )
    logger.info(
        f"Training solely on the preference data {args.dpo_type}, dataset size: {len(pref_data): d}, optimal reward: {formatted_opt_reward}, reward: {formatted_reward}, reward error: {formatted_rew_error}."
    )
    rew_err_dict, rew_dict = dict(), dict()
    rew_err_dict[args.pref_data_num] = rew_error
    rew_dict[args.pref_data_num] = reward
    save_path = os.path.join(log_dir, f"reward_error_{args.dpo_type}.yml")
    yaml.dump(rew_err_dict, open(save_path, "w"), default_flow_style=False)
    save_path = os.path.join(log_dir, f"reward_{args.dpo_type}.yml")
    yaml.dump(rew_dict, open(save_path, "w"), default_flow_style=False)


    #calculating errors if param is known
    known_param_rewards=[]
    known_param_rew_err=[]
    for i in range(group_num):
        reward=env.evaluate_reward_group_wise(policy=ret_policy(action_num,policy_feature_func,reward_param[i]),states=test_pref)
        reward_err=[float((a-b)/a) for a,b in zip(opt_reward,reward)]
        known_param_rewards.append(reward)
        known_param_rew_err.append(reward_err)
    print(known_param_rewards)
    #formatted_known_param_rewards  = ", ".join([f"{reward:.4f}" for reward in known_param_rewards])
    #known_param_rew_err=[float((a-b)/a) for a,b in zip(opt_reward,known_param_rewards)]
    #formatted_known_param_rew_err  = ", ".join([f"{err:.4f}" for err in known_param_rew_err])
    logger.info(
        f"optimal reward: {formatted_opt_reward}, known_param_reward: {known_param_rewards}, Known param reward error: {known_param_rew_err}."
    )
    if args.wandb_use:
        d_wandb = {}
        # Assuming rew_err is a list
        for i, err in enumerate(rew_error):
            key = f"final/reward_err_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
            d_wandb[key] = err
        for i, param in enumerate(policy_param):
            key = f"final/reward_param_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
            d_wandb[key] = param      
        for i, opt_r in enumerate(opt_reward):
            key = f"optimal_reward_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
            d_wandb[key] = opt_r 
        for i, rew in enumerate(reward):
            key = f"final/reward_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
            d_wandb[key] = rew  
        for i, rew in enumerate(known_param_rewards):
            for j, r in enumerate(rew):
                key = f"reward_{j}_when_{i + 1}_group_param_known"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = r
        for i, err in enumerate(known_param_rew_err):
             for j, e in enumerate(err):
                key = f"reward_error_{j}_when_{i + 1}_group_param_known"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = e 
        
        wandb.log(d_wandb)

        wandb.finish()

    """
    # RMB-PO
    logger.info(
        f"Train a policy on the preference data with policy-generated data (RMB-PO)."
    )
    rl_data = pref_to_rl(pref_data)
    agent = PolicyGradient(
        policy_feature_func,
        learned_reward_func,
        uniform_policy,
        feature_dim,
        action_num,
        args.reg_coef,
        args.pg_step_size,
        args.pg_num_iters,
        args.pg_adaptive,
        args.pg_ada_coef,
        logger=logger,
    )

    reward = agent.train(dataset=rl_data, env=env, learned_env=learned_env)
    rew_error = float(opt_reward - reward)
    policy_param = agent.get_param
    logger.info(f"Policy parameter (RMB-PO): {policy_param}.")
    logger.info(
        f"Training solely on the preference data (RMB-PO), dataset size: {len(rl_data): d}, optimal reward: {opt_reward: .4f}, reward: {reward: .4f}, reward error: {rew_error: .4f}."
    )
    rew_err_dict, rew_dict = dict(), dict()
    rew_err_dict[args.pref_data_num] = rew_error
    rew_dict[args.pref_data_num] = float(reward)
    save_path = os.path.join(log_dir, "reward_error_rmb_po.yml")
    yaml.dump(rew_err_dict, open(save_path, "w"), default_flow_style=False)
    save_path = os.path.join(log_dir, "reward_rmb_po.yml")
    yaml.dump(rew_dict, open(save_path, "w"), default_flow_style=False)

    # RMB-PO+: Collect a new RL data
    logger.info(f"Train a policy on the augmented data (RMB-PO+).")
    new_rl_data_num = int(args.pref_data_num * args.rl_data_ratio)
    new_rl_data = collect_rl_data(new_rl_data_num, env)
    aug_rl_data = merge_datasets(pref_data, new_rl_data)
    agent = PolicyGradient(
        policy_feature_func,
        learned_reward_func,
        uniform_policy,
        feature_dim,
        action_num,
        args.reg_coef,
        args.pg_step_size,
        args.pg_num_iters,
        args.pg_adaptive,
        args.pg_ada_coef,
        logger=logger,
    )
    reward = agent.train(aug_rl_data, env, learned_env)
    policy_param = agent.get_param
    logger.info(
        f"Policy parameter learned on the augmented data (RMB-PO+): {policy_param}."
    )
    rew_error = float(opt_reward - reward)
    logger.info(
        f"Training on the augmented data (RMB-PO+), augmented dataset size: {len(aug_rl_data): d}, optimal reward: {opt_reward: .4f}, reward: {reward: .4f}, reward error: {rew_error: .4f}."
    )
    rew_err_dict, rew_dict = dict(), dict()
    rew_err_dict[args.pref_data_num] = rew_error
    rew_dict[args.pref_data_num] = float(reward)
    save_path = os.path.join(log_dir, "reward_error_rmb_po_plus.yml")
    yaml.dump(rew_err_dict, open(save_path, "w"), default_flow_style=False)
    save_path = os.path.join(log_dir, "reward_aug_rmb_po_plus.yml")
    yaml.dump(rew_dict, open(save_path, "w"), default_flow_style=False)
    """

if __name__ == "__main__":
    main(parse_args())
