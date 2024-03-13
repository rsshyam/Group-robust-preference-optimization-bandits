from collections import defaultdict
import copy
import cvxpy as cp
import numpy as np
from typing import List,Union
from envs.group_linear_bandit import GroupLinearBandit
from utils.collect_data import GroupTransition, ret_uniform_policy, collect_preference_data
from utils.utils import softmax, softmax_2D, sigmoid
from utils.logger import Logger
import wandb

class GroupDirectPolicyOptimizationVectorised:
    """ Group DPO/IPO. Includes justdpo (DPO), linear (IPO), and log (IPO).
    The DPO implementation is *vectorised* (unlike class GroupDirectPolicyOptimization).
    The DPO loss and grad are calculated using the known policy 
    π = exp(r)/sum(exp(r_all)), for r = <φ(s,a,g), θ> (vectorisation).
    """

    def __init__(
        self,
        state_dim: int,                       ## state s drawn as vector of `state_dim` elements from Uniform(0,1)
        action_num: int,                      ## number of actions in Action Space
        group_num: int,                       ## number of groups
        feature_dim: int,                     ## feature_dim = 2 * state_dim (num elements in vector φ(s,a,g) )
        feature_func,                         ## φ(s,a,g)
        ref_policy,                           ## π_ref(a|s)
        reg_coef: float,                      ## β scaling in the DPO gradient & loss -- controls KL Divergence from π_ref
        step_size: float,                     ## η_θ step size for Gradient Descent on the DPO/IPO loss (if not is_adaptive)
        num_iters: int,                       ## number of update steps on Training dataset
        is_adaptive: bool = False,            ## if is_adaptive, step size in Update step is adaptive to the historical grad
        ada_coef: float = None,               ## coef scaling the inverted-sqrt historical grad in Update step if is_adaptive
        logger: Logger = None,                ## logger
        wandb_use: bool = False,              ## recording results in WandB
        ipo_grad_type: str = 'justdpo',       ## `justdpo` (vectorised version), `linear` (IPO), or `log` (IPO)
        param_limit: int = 1,                 ## elements of vector θ range in [0, param_limit]
        lamba: float=0,                       ## L2 regularisation for closed-form regression of IPO objective in Linear Bandits case
        train_agent: bool=True                ## if True, use self.train(); else, use self.random_train() func
    ) -> None:
        self.state_dim = state_dim
        self.action_num = action_num
        self.group_num = group_num
        self.feature_dim = feature_dim
        self.feature_func = feature_func
        self.ref_policy = ref_policy
        self.reg_coef = reg_coef
        self.step_size = step_size
        self.num_iters = num_iters
        self.logger = logger
        self.wandb_use = wandb_use
        self.ipo_grad_type = ipo_grad_type
        
        self.hist_group_loss=np.zeros(group_num)
        self.group_loss=np.zeros(group_num)

        self.is_adaptive = is_adaptive
        self.ada_coef = ada_coef
        self.hist_grad_squared_norm = 0.0

        # initialize the learnt policy parameter θ -- same param for all groups g
        self.param = np.random.uniform(0, param_limit, self.feature_dim)
        self.lamba = lamba
        self.train_agent=train_agent

        print('Vectorised DPO; step size = ', self.step_size)

    def ret_action_prob(self, state: np.ndarray, group_id: int) -> np.ndarray:
        arr = np.zeros(self.action_num, np.float32)
        for action_idx in range(self.action_num):
            feature = self.feature_func(state, action_idx, group_id) # (num_states, state_dim*2=feature_dim)
            arr[action_idx] = np.dot(feature, self.param)
        prob = softmax(arr)
        return prob

    def ret_policy(self):
        action_num = self.action_num
        feature_func = copy.deepcopy(self.feature_func)
        param = self.param

        def policy(state: np.ndarray, group_id: int) -> np.ndarray:
            arr = np.zeros(action_num, np.float32)
            for action_idx in range(action_num):
                feature = feature_func(state, action_idx, group_id)
                arr[action_idx] = np.dot(feature, param)
            prob = softmax(arr)

            return prob

        return policy

    def sample_action(self, state: np.ndarray, group_id: int) -> int:
        prob = self.action_prob(state, group_id)
        sampled_act = np.random.choice(a=self.action_num, size=1, replace=True, p=prob)
        return sampled_act
    
    def update_once_nonvectorised(self, dataset: List[GroupTransition]) -> float:
        grad = np.zeros_like(self.param)
        
        group_loss=np.zeros(self.group_num)
        cur_group_counts=np.zeros(self.group_num)
        
        for transition in dataset:
            state, action_one, action_two, group_id, pref = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.group_id,
                transition.pref,
            )
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one

            feat_pref_act, feat_non_pref_act = (
                self.feature_func(state, pref_act, group_id),
                self.feature_func(state, non_pref_act,group_id),
            )
            #print(feat_pref_act,feat_non_pref_act,self.param)
            cur_policy_act_prob = self.ret_action_prob(state,group_id)
            ref_policy_act_prob = self.ref_policy(state,group_id)
            
            if self.ipo_grad_type=='justdpo':
                log_ratio_diff = self.reg_coef * (feat_pref_act - feat_non_pref_act) @ (self.param) # VECTORISED REDEFINITION instead of log-sum
                coef = sigmoid(-log_ratio_diff)
                group_loss[group_id] += -np.log(sigmoid(log_ratio_diff))#+self.adj[group_id]/np.sqrt(self.group_counts[group_id]) #calculate group losses
            elif self.ipo_grad_type=='linear':
                lin_diff = (feat_pref_act - feat_non_pref_act) @ (self.param) - 0.5*(1/self.reg_coef)
                coef = -2*lin_diff/self.reg_coef
                group_loss[group_id] += np.square(lin_diff)#+self.adj[group_id]/np.sqrt(self.group_counts[group_id])
            elif self.ipo_grad_type=='log':
                log_diff = (
                    np.log((cur_policy_act_prob[pref_act]*ref_policy_act_prob[non_pref_act])/(cur_policy_act_prob[non_pref_act]*ref_policy_act_prob[pref_act])+1e-6 )
                )
                coef = -2*(log_diff-0.5*(1/self.reg_coef))/self.reg_coef
                group_loss[group_id] += np.square((log_diff-0.5*(1/self.reg_coef)))#+self.adj[group_id]/np.sqrt(self.group_counts[group_id])
            else:
                raise ValueError('value not implemented')
            
            #print(group_id,self.adj[group_id]/np.sqrt(self.group_counts[group_id]) )
            cur_group_counts[group_id] += 1
            
            neg_cur_data_grad = (
                self.reg_coef * coef * (feat_pref_act - feat_non_pref_act)
            )

            grad -= neg_cur_data_grad

        grad /= len(dataset)

        group_loss=group_loss/cur_group_counts
        #print(group_loss)
        self.hist_grad_squared_norm += np.sum(np.square(grad))
        self.hist_group_loss += group_loss
        self.group_loss = group_loss
        if self.is_adaptive:
            step_size = self.ada_coef / np.sqrt(self.hist_grad_squared_norm)
        else:
            step_size = self.step_size
        self.param = self.param - step_size * grad
        return np.sqrt(np.sum(np.square(grad))) # grad L2-norm

    def update_once(self, dataset: List[GroupTransition]) -> float:
        grad = np.zeros_like(self.param)
        
        group_loss=np.zeros(self.group_num)
        cur_group_counts=np.zeros(self.group_num)

        group_id_idx_all = defaultdict(list)
        feature_diff_all = np.zeros((len(dataset), self.feature_dim))
        pref_act_all = []
        non_pref_act_all = []
        cur_policy_act_prob_all = np.zeros((len(dataset), self.action_num))
        ref_policy_act_prob_all = np.zeros((len(dataset), self.action_num))
        
        for idx, transition in enumerate(dataset):
            state, action_one, action_two, group_id, pref = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.group_id,
                transition.pref,
            )
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one

            pref_act_all.append(pref_act)
            non_pref_act_all.append(non_pref_act)

            feat_pref_act, feat_non_pref_act = (
                self.feature_func(state, pref_act, group_id),
                self.feature_func(state, non_pref_act,group_id),
            )
            feature_diff_all[idx,:] = feat_pref_act - feat_non_pref_act
            
            cur_policy_act_prob_all[idx,:] = self.ret_action_prob(state,group_id)
            ref_policy_act_prob_all[idx,:] = self.ref_policy(state,group_id)

            group_id_idx_all[group_id].append(idx) # get dataset indices for each group
            cur_group_counts[group_id] += 1

        ######################################################################################
        ################### VECTORISED REDEFINITION across all transitions ###################
        ######################################################################################
        if self.ipo_grad_type=='justdpo':
            log_ratio_diff_all = self.reg_coef * feature_diff_all @ self.param.reshape(self.feature_dim,1) # log_ratio_diff_all shape (len(dataset),1)
            coef = sigmoid(-log_ratio_diff_all) # shape (len(dataset),1)
            for group_id in range(self.group_num):
                group_indices = group_id_idx_all[group_id]
                group_loss[group_id] = np.sum(-np.log(sigmoid(log_ratio_diff_all[group_indices])))#+self.adj[group_id]/np.sqrt(self.group_counts[group_id]) #calculate group losses
        elif self.ipo_grad_type=='linear':
            lin_diff = feature_diff_all @ self.param.reshape(self.feature_dim,1) - 0.5*(1/self.reg_coef)
            coef = -2*lin_diff/self.reg_coef
            for group_id in range(self.group_num):
                group_indices = group_id_idx_all[group_id]
                group_loss[group_id] = np.sum(np.square(lin_diff[group_indices]))#+self.adj[group_id]/np.sqrt(self.group_counts[group_id])
        elif self.ipo_grad_type=='log':
            row_indices = np.arange(cur_policy_act_prob_all.shape[0])
            log_diff = (
                np.log((cur_policy_act_prob_all[row_indices,pref_act_all]*ref_policy_act_prob_all[row_indices,non_pref_act])/
                       (cur_policy_act_prob_all[row_indices,non_pref_act]*ref_policy_act_prob_all[row_indices,pref_act_all])+1e-6 )
            )
            coef = -2*(log_diff-0.5*(1/self.reg_coef))/self.reg_coef
            for group_id in range(self.group_num):
                group_indices = group_id_idx_all[group_id]
                group_loss[group_id] = np.sum(np.square((log_diff[group_indices]-0.5*(1/self.reg_coef))))#+self.adj[group_id]/np.sqrt(self.group_counts[group_id])
        else:
            raise ValueError('value not implemented')

        neg_cur_data_grad = self.reg_coef * coef * feature_diff_all
        grad = np.sum(-neg_cur_data_grad, axis=0) / len(dataset)

        group_loss /= cur_group_counts

        self.hist_grad_squared_norm += np.sum(np.square(grad))
        self.hist_group_loss += group_loss
        self.group_loss = group_loss
        if self.is_adaptive:
            step_size = self.ada_coef / np.sqrt(self.hist_grad_squared_norm)
        else:
            step_size = self.step_size
        self.param = self.param - step_size * grad
        return np.sqrt(np.sum(np.square(grad))) # grad L2-norm
    
    def evaluate_ipo_loss(self, dataset: List[GroupTransition], policy=None) -> float:
        """
        Evaluate the loss on the dataset for any policy.
        """
        if policy is None:
            policy = self.ret_policy()

        feature_diff_all = np.zeros((len(dataset), self.feature_dim))
        pref_act_all = []
        non_pref_act_all = []
        eval_policy_act_prob_all = np.zeros((len(dataset), self.action_num))
        ref_policy_act_prob_all = np.zeros((len(dataset), self.action_num))

        for idx, transition in enumerate(dataset):
            state, action_one, action_two, group_id, pref = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.group_id,
                transition.pref,
            )
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one

            pref_act_all.append(pref_act)
            non_pref_act_all.append(non_pref_act)

            feat_pref_act, feat_non_pref_act = (
                self.feature_func(state, pref_act, group_id),
                self.feature_func(state, non_pref_act,group_id),
            )
            feature_diff_all[idx,:] = feat_pref_act - feat_non_pref_act
            
            eval_policy_act_prob_all[idx,:] = policy(state,group_id)
            ref_policy_act_prob_all[idx,:] = self.ref_policy(state,group_id)

        if self.ipo_grad_type=='linear':
            lin_diff = feature_diff_all @ self.param.reshape(self.feature_dim,1) - 0.5*(1/self.reg_coef)
            coef = lin_diff
        elif self.ipo_grad_type=='log':
            row_indices = np.arange(eval_policy_act_prob_all.shape[0])
            log_diff=(
                np.log((eval_policy_act_prob_all[row_indices,pref_act_all]*ref_policy_act_prob_all[row_indices,non_pref_act_all])/
                       (eval_policy_act_prob_all[row_indices,non_pref_act_all]*ref_policy_act_prob_all[row_indices,pref_act_all]) + 1e-6)
            )
            coef=(log_diff-0.5*(1/self.reg_coef))
        else: # self.ipo_grad_type=='linear'
            lin_diff = feature_diff_all @ self.param.reshape(self.feature_dim,1) - 0.5*(1/self.reg_coef)
            coef = lin_diff

        loss = np.sum(np.square(coef)) / len(dataset)
        return loss
    
    def evaluate_ipo_grad(self, dataset: List[GroupTransition], policy=None) -> float:
        """
        Evaluate the loss on the dataset for any policy.
        """
        if policy is None:
            policy = self.ret_policy()

        grad = np.zeros_like(self.param)
        
        feature_diff_all = np.zeros((len(dataset), self.feature_dim))
        pref_act_all = []
        non_pref_act_all = []
        eval_policy_act_prob_all = np.zeros((len(dataset), self.action_num))
        ref_policy_act_prob_all = np.zeros((len(dataset), self.action_num))

        for idx, transition in enumerate(dataset):
            state, action_one, action_two, group_id, pref = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.group_id,
                transition.pref,
            )
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one

            pref_act_all.append(pref_act)
            non_pref_act_all.append(non_pref_act)

            feat_pref_act, feat_non_pref_act = (
                self.feature_func(state, pref_act, group_id),
                self.feature_func(state, non_pref_act,group_id),
            )
            feature_diff_all[idx,:] = feat_pref_act - feat_non_pref_act
            
            eval_policy_act_prob_all[idx,:] = policy(state,group_id)
            ref_policy_act_prob_all[idx,:] = self.ref_policy(state,group_id)

        if self.ipo_grad_type=='linear':
            lin_diff = feature_diff_all @ self.param.reshape(self.feature_dim,1) - 0.5*(1/self.reg_coef)
            coef = lin_diff
        elif self.ipo_grad_type=='log':
            row_indices = np.arange(eval_policy_act_prob_all.shape[0])
            log_diff=(
                np.log((eval_policy_act_prob_all[row_indices,pref_act_all]*ref_policy_act_prob_all[row_indices,non_pref_act_all])/
                       (eval_policy_act_prob_all[row_indices,non_pref_act_all]*ref_policy_act_prob_all[row_indices,pref_act_all]) + 1e-6)
            )
            coef=(log_diff-0.5*(1/self.reg_coef))
        else:
            print(self.param,feat_pref_act-feat_non_pref_act)
            lin_diff = feature_diff_all @ self.param.reshape(self.feature_dim,1) - 0.5*(1/self.reg_coef)
            coef = lin_diff
        cur_data_grad = 2 * coef * feature_diff_all

        grad = np.sum(cur_data_grad, axis=0) / len(dataset)
        return np.sqrt(np.sum(np.square(grad)))
    
    def evaluate_ipo_grp_loss(self, dataset: List[GroupTransition], policy=None) -> float:
        """
        Evaluate the loss on the dataset for any policy.
        """
        if policy is None:
            policy = self.ret_policy()

        loss = np.zeros(self.group_num)
        counts = np.zeros(self.group_num)
        
        group_id_idx_all = defaultdict(list)
        feature_diff_all = np.zeros((len(dataset), self.feature_dim))
        pref_act_all = []
        non_pref_act_all = []
        eval_policy_act_prob_all = np.zeros((len(dataset), self.action_num))
        ref_policy_act_prob_all = np.zeros((len(dataset), self.action_num))

        for idx, transition in enumerate(dataset):
            state, action_one, action_two, group_id, pref = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.group_id,
                transition.pref,
            )
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one

            pref_act_all.append(pref_act)
            non_pref_act_all.append(non_pref_act)

            feat_pref_act, feat_non_pref_act = (
                self.feature_func(state, pref_act, group_id),
                self.feature_func(state, non_pref_act,group_id),
            )
            feature_diff_all[idx,:] = feat_pref_act - feat_non_pref_act
            
            eval_policy_act_prob_all[idx,:] = policy(state,group_id)
            ref_policy_act_prob_all[idx,:] = self.ref_policy(state,group_id)

            group_id_idx_all[group_id].append(idx) # get dataset indices for each group
            counts[group_id] += 1

        if self.ipo_grad_type=='linear':
            lin_diff = feature_diff_all @ self.param.reshape(self.feature_dim,1) - 0.5*(1/self.reg_coef)
            coef = lin_diff
        elif self.ipo_grad_type=='log':
            row_indices = np.arange(eval_policy_act_prob_all.shape[0])
            log_diff=(
                np.log((eval_policy_act_prob_all[row_indices,pref_act_all]*ref_policy_act_prob_all[row_indices,non_pref_act_all])/
                       (eval_policy_act_prob_all[row_indices,non_pref_act_all]*ref_policy_act_prob_all[row_indices,pref_act_all]) + 1e-6)
            )
            coef=(log_diff-0.5*(1/self.reg_coef))
        else: # self.ipo_grad_type=='linear'
            lin_diff = feature_diff_all @ self.param.reshape(self.feature_dim,1) - 0.5*(1/self.reg_coef)
            coef = lin_diff

        for group_id in range(self.group_num):
            group_indices = group_id_idx_all[group_id]
            loss[group_id] = np.sum(np.square(coef[group_indices]))

        loss = loss/counts
        return loss
    
    def evaluate_grp_loss(self, dataset: List[GroupTransition], policy=None) -> float:
        """
        Evaluate the loss on the dataset for any policy.
        """
        if policy is None:
            policy = self.ret_policy()

        loss = np.zeros(self.group_num)
        counts = np.zeros(self.group_num)
        group_id_idx_all = defaultdict(list)
        feature_diff_all = np.zeros((len(dataset), self.feature_dim))

        for idx, transition in enumerate(dataset):
            state, action_one, action_two, group_id, pref = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.group_id,
                transition.pref,
            )
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one

            feat_pref_act, feat_non_pref_act = (
                self.feature_func(state, pref_act, group_id),
                self.feature_func(state, non_pref_act,group_id),
            )
            feature_diff_all[idx,:] = feat_pref_act - feat_non_pref_act
            
            group_id_idx_all[group_id].append(idx) # get dataset indices for each group
            counts[group_id] += 1
        
        # VECTORISATION for log_ratio_diff
        log_ratio_diff = self.reg_coef * feature_diff_all @ self.param.reshape(self.feature_dim,1)

        for group_id in range(self.group_num):
            group_indices = group_id_idx_all[group_id]
            loss[group_id] = np.sum(-np.log(sigmoid(log_ratio_diff[group_indices])))
        
        loss = loss/counts
        return loss
    
    def evaluate_loss(self, dataset: List[GroupTransition], policy=None) -> float:
        """
        Evaluate the loss on the dataset for any policy.
        """
        if policy is None:
            policy = self.ret_policy()

        loss = 0.0
        feature_diff_all = np.zeros((len(dataset), self.feature_dim))

        for idx, transition in enumerate(dataset):
            state, action_one, action_two, group_id, pref = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.group_id,
                transition.pref,
            )
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one

            feat_pref_act, feat_non_pref_act = (
                self.feature_func(state, pref_act, group_id),
                self.feature_func(state, non_pref_act,group_id),
            )
            feature_diff_all[idx,:] = feat_pref_act - feat_non_pref_act
                    
        # VECTORISATION for log_ratio_diff
        log_ratio_diff = self.reg_coef * feature_diff_all @ self.param.reshape(self.feature_dim,1)

        loss = np.sum(-np.log(sigmoid(log_ratio_diff))) / len(dataset)
        return loss
    
    def Regression(self, dataset: List[GroupTransition],lamba: float)-> float:
        Y=[]
        group_id_mat=[]
        for transition in dataset:
            state, action_one, action_two, group_id, pref = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.group_id,
                transition.pref,
            )
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one
            feat_pref_act, feat_non_pref_act = (
                self.feature_func(state, pref_act, group_id),
                self.feature_func(state, non_pref_act,group_id),
            )
            Y.append(feat_pref_act-feat_non_pref_act)
            group_id_mat.append(group_id)
        Y=np.array(Y)
        group_id_mat=np.array(group_id_mat)
        print(Y,Y.shape)
        coef=np.linalg.inv(Y.transpose()@Y/self.group_num+lamba*np.eye(Y.shape[1]))
        print(np.linalg.det(np.matmul(Y.transpose(),Y)))
        variate=np.matmul(Y.transpose()/self.group_num,np.ones([len(dataset),1]))
        self.param=np.matmul(coef,variate).ravel()/(2*self.reg_coef)
        
        unique_groups = np.unique(group_id_mat)

        for group_id in unique_groups:
            group_indices = np.where(group_id_mat == group_id)[0]
            Y_group = Y[group_indices, :]

            # Perform the calculation for each group
            result_group = np.square(np.dot(Y_group, self.param) - 1/(2*self.reg_coef))

            result_group_avg=np.mean(result_group)
            print(result_group_avg.shape)
            # Append the result for this group to the overall result
            self.group_loss[group_id]=result_group_avg
        live_grad=((Y@self.param).T-1/(2*self.reg_coef)).dot(Y)+lamba*self.param
        return np.sqrt(np.sum(np.square(live_grad)))
    
    def random_train(self, dataset: List[GroupTransition],
              val_dataset: list[GroupTransition],
              test_dataset: list[GroupTransition],  env: GroupLinearBandit, optimal_reward: List[float]) -> float:
            
        grad_norm=self.evaluate_ipo_grad(dataset)
        live_grad=grad_norm
        train_loss=self.evaluate_ipo_loss(dataset)
        val_loss = self.evaluate_ipo_loss(val_dataset)

        train_grp_loss = self.evaluate_ipo_grp_loss(dataset)
        val_grp_loss = self.evaluate_ipo_grp_loss(val_dataset)
                    
        kl_dist=self.evaluate_KL(env=env,states=test_dataset)

        formatted_kl=", ".join([f"{kld:.4f}" for kld in kl_dist])

        #Evaluate the reward on the test dataset:
        #print(optimal_reward,self.evaluate_reward(env=env, 
        #                           states=test_dataset))
        rew_err = [float(a - b)/a for a, b in zip(optimal_reward,self.evaluate_reward(env=env, 
                                    states=test_dataset) )]
        formatted_rew=", ".join([f"{reward:.4f}" for reward in rew_err])

        max_reward_err=max(rew_err)
        max_reward_err_index=rew_err.index(max_reward_err)

        max_kl_dist=max(kl_dist)
        max_kl_dist_index=kl_dist.index(max_kl_dist)

        max_train_grp_loss=np.max(train_grp_loss)
        max_val_grp_loss=np.max(val_grp_loss)
        max_train_grp_loss_index=np.argmax(train_grp_loss)
        max_val_grp_loss_index=np.argmax(val_grp_loss)
        
        
        step=0
        logging_str = (f"Iteration: {step: d}, train_loss: {train_loss: .4f}, "
                    f"val_loss: {val_loss: .4f}, grad_norm: {grad_norm:.4f}, live_grad: {live_grad:.4f}, "
                    f"reward_err: {formatted_rew}, KL_dist: {formatted_kl}, param: {self.param}, "
                    f"train_grp_loss: {train_grp_loss}, val_grp_loss: {val_grp_loss}, "
                    f"max_reward_err: {max_reward_err: .4f}, max_reward_err_index: {max_reward_err_index}, "
                    f"max_kl_dist: {max_kl_dist: .4f}, max_kl_dist_index: {max_kl_dist_index}, "
                    f"max_train_grp_loss: {max_train_grp_loss: .4f}, max_train_grp_loss_index: {max_train_grp_loss_index}, "
                    f"max_val_grp_loss: {max_val_grp_loss: .4f}, max_val_grp_loss_index: {max_val_grp_loss_index}, ")
        
        if self.wandb_use:
            d_wandb = {
                "Iteration": step, "train_loss": train_loss, 
                    "val_loss": val_loss, "grad_norm": grad_norm, "live_grad": live_grad,
                    "max_reward_err": max_reward_err , "max_reward_err_index": max_reward_err_index, 
                    "max_kl_dist" : max_kl_dist, "max_kl_dist_index": max_kl_dist_index, 
                    "max_train_grp_loss": max_train_grp_loss, "max_train_grp_loss_index": max_train_grp_loss_index, 
                    "max_val_grp_loss": max_val_grp_loss, "max_val_grp_loss_index": max_val_grp_loss_index, 
            }
            # Assuming rew_err is a list
            for i, err in enumerate(rew_err):
                key = f"reward_err_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = err
            for i, param in enumerate(self.param):
                key = f"reward_param_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = param 
            for i, grp_ls in enumerate(train_grp_loss):
                key = f"train_group_loss_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = grp_ls    
            for i, grp_ls in enumerate(val_grp_loss):
                key = f"val_group_loss_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = grp_ls   
            for i, kld in enumerate(kl_dist):
                key = f"KL_distance_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = kld 
            wandb.log(d_wandb)
        
        if self.logger:
            self.logger.info(logging_str)
        else:
            print(logging_str)
    
    
        rew = self.evaluate_reward(env, test_dataset)
        return rew
    
    
    
    def train(self, dataset: List[GroupTransition],
              val_dataset: list[GroupTransition],
              test_dataset: list[GroupTransition],  env: GroupLinearBandit, optimal_reward: List[float]) -> float:
        print('ipo grad type: ', self.ipo_grad_type)
        if self.ipo_grad_type=='Regression':
            print('Hellooooo Regression IPO')
            """
            grad_norm=self.evaluate_ipo_grad(dataset)
            live_grad=grad_norm
            train_loss=self.evaluate_ipo_loss(dataset)
            val_loss = self.evaluate_ipo_loss(val_dataset)

            train_grp_loss = self.evaluate_ipo_grp_loss(dataset)
            val_grp_loss = self.evaluate_ipo_grp_loss(val_dataset)
                        
            kl_dist=self.evaluate_KL(env=env,states=test_dataset)

            formatted_kl=", ".join([f"{kld:.4f}" for kld in kl_dist])

            #Evaluate the reward on the test dataset:
            #print(optimal_reward,self.evaluate_reward(env=env, 
            #                           states=test_dataset))
            rew_err = [float(a - b)/a for a, b in zip(optimal_reward,self.evaluate_reward(env=env, 
                                        states=test_dataset) )]
            formatted_rew=", ".join([f"{reward:.4f}" for reward in rew_err])

            max_reward_err=max(rew_err)
            max_reward_err_index=rew_err.index(max_reward_err)

            max_kl_dist=max(kl_dist)
            max_kl_dist_index=kl_dist.index(max_kl_dist)

            max_train_grp_loss=np.max(train_grp_loss)
            max_val_grp_loss=np.max(val_grp_loss)
            max_train_grp_loss_index=np.argmax(train_grp_loss)
            max_val_grp_loss_index=np.argmax(val_grp_loss)
            
            
            step=-1
            logging_str = (f"Iteration: {step: d}, train_loss: {train_loss: .4f}, "
                        f"val_loss: {val_loss: .4f}, grad_norm: {grad_norm:.4f}, live_grad: {live_grad:.4f}, "
                        f"reward_err: {formatted_rew}, KL_dist: {formatted_kl}, param: {self.param}, "
                        f"train_grp_loss: {train_grp_loss}, val_grp_loss: {val_grp_loss}, "
                        f"max_reward_err: {max_reward_err: .4f}, max_reward_err_index: {max_reward_err_index}, "
                        f"max_kl_dist: {max_kl_dist: .4f}, max_kl_dist_index: {max_kl_dist_index}, "
                        f"max_train_grp_loss: {max_train_grp_loss: .4f}, max_train_grp_loss_index: {max_train_grp_loss_index}, "
                        f"max_val_grp_loss: {max_val_grp_loss: .4f}, max_val_grp_loss_index: {max_val_grp_loss_index}, ")
            
            if self.wandb_use:
                d_wandb = {
                    "Iteration": step, "train_loss": train_loss, 
                        "val_loss": val_loss, "grad_norm": grad_norm, "live_grad": live_grad,
                        "max_reward_err": max_reward_err , "max_reward_err_index": max_reward_err_index, 
                        "max_kl_dist" : max_kl_dist, "max_kl_dist_index": max_kl_dist_index, 
                        "max_train_grp_loss": max_train_grp_loss, "max_train_grp_loss_index": max_train_grp_loss_index, 
                        "max_val_grp_loss": max_val_grp_loss, "max_val_grp_loss_index": max_val_grp_loss_index, 
                }
                # Assuming rew_err is a list
                for i, err in enumerate(rew_err):
                    key = f"reward_err_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                    d_wandb[key] = err
                for i, param in enumerate(self.param):
                    key = f"reward_param_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                    d_wandb[key] = param 
                for i, grp_ls in enumerate(train_grp_loss):
                    key = f"train_group_loss_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                    d_wandb[key] = grp_ls    
                for i, grp_ls in enumerate(val_grp_loss):
                    key = f"val_group_loss_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                    d_wandb[key] = grp_ls   
                for i, kld in enumerate(kl_dist):
                    key = f"KL_distance_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                    d_wandb[key] = kld 
                wandb.log(d_wandb)
            
            if self.logger:
                self.logger.info(logging_str)
            else:
                print(logging_str)
            """
            live_grad=self.Regression(dataset,lamba=self.lamba)
            grad_norm=self.evaluate_ipo_grad(dataset)
            train_loss=self.evaluate_ipo_loss(dataset)
            val_loss = self.evaluate_ipo_loss(val_dataset)

            train_grp_loss = self.evaluate_ipo_grp_loss(dataset)
            val_grp_loss = self.evaluate_ipo_grp_loss(val_dataset)
                        
            kl_dist=self.evaluate_KL(env=env,states=test_dataset)

            formatted_kl=", ".join([f"{kld:.4f}" for kld in kl_dist])

            #Evaluate the reward on the test dataset:
            #print(optimal_reward,self.evaluate_reward(env=env, 
            #                           states=test_dataset))
            rew_err = [float(a - b)/a for a, b in zip(optimal_reward,self.evaluate_reward(env=env, 
                                        states=test_dataset) )]
            formatted_rew=", ".join([f"{reward:.4f}" for reward in rew_err])

            max_reward_err=max(rew_err)
            max_reward_err_index=rew_err.index(max_reward_err)

            max_kl_dist=max(kl_dist)
            max_kl_dist_index=kl_dist.index(max_kl_dist)

            max_train_grp_loss=np.max(train_grp_loss)
            max_val_grp_loss=np.max(val_grp_loss)
            max_train_grp_loss_index=np.argmax(train_grp_loss)
            max_val_grp_loss_index=np.argmax(val_grp_loss)
        
            step=0
            logging_str = (f"Iteration: {step: d}, train_loss: {train_loss: .4f}, "
                        f"val_loss: {val_loss: .4f}, grad_norm: {grad_norm:.4f}, live_grad: {live_grad:.4f}, "
                        f"reward_err: {formatted_rew}, KL_dist: {formatted_kl}, param: {self.param}, "
                        f"train_grp_loss: {train_grp_loss}, val_grp_loss: {val_grp_loss}, "
                        f"max_reward_err: {max_reward_err: .4f}, max_reward_err_index: {max_reward_err_index}, "
                        f"max_kl_dist: {max_kl_dist: .4f}, max_kl_dist_index: {max_kl_dist_index}, "
                        f"max_train_grp_loss: {max_train_grp_loss: .4f}, max_train_grp_loss_index: {max_train_grp_loss_index}, "
                        f"max_val_grp_loss: {max_val_grp_loss: .4f}, max_val_grp_loss_index: {max_val_grp_loss_index}, ")
            
            if self.wandb_use:
                d_wandb = {
                    "Iteration": step, "train_loss": train_loss, 
                        "val_loss": val_loss, "grad_norm": grad_norm, "live_grad": live_grad,
                        "max_reward_err": max_reward_err , "max_reward_err_index": max_reward_err_index, 
                        "max_kl_dist" : max_kl_dist, "max_kl_dist_index": max_kl_dist_index, 
                        "max_train_grp_loss": max_train_grp_loss, "max_train_grp_loss_index": max_train_grp_loss_index, 
                        "max_val_grp_loss": max_val_grp_loss, "max_val_grp_loss_index": max_val_grp_loss_index, 
                }
                # Assuming rew_err is a list
                for i, err in enumerate(rew_err):
                    key = f"reward_err_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                    d_wandb[key] = err
                for i, param in enumerate(self.param):
                    key = f"reward_param_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                    d_wandb[key] = param 
                for i, grp_ls in enumerate(train_grp_loss):
                    key = f"train_group_loss_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                    d_wandb[key] = grp_ls    
                for i, grp_ls in enumerate(val_grp_loss):
                    key = f"val_group_loss_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                    d_wandb[key] = grp_ls   
                for i, kld in enumerate(kl_dist):
                    key = f"KL_distance_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                    d_wandb[key] = kld 
                wandb.log(d_wandb)
            
            if self.logger:
                self.logger.info(logging_str)
            else:
                print(logging_str)
            rew = self.evaluate_reward(env, test_dataset)
            return rew
        for step in range(self.num_iters):
            grad_norm = self.update_once(dataset)
            if step % 2000 == 0:
                if self.ipo_grad_type=='justdpo':
                    train_loss = self.evaluate_loss(dataset)
                    val_loss = self.evaluate_loss(val_dataset)
                else:
                    train_loss = self.evaluate_ipo_loss(dataset)
                    val_loss = self.evaluate_ipo_loss(val_dataset)


                if self.ipo_grad_type=='justdpo':
                    train_grp_loss = self.evaluate_grp_loss(dataset)
                    val_grp_loss = self.evaluate_grp_loss(val_dataset)
                else:
                    train_grp_loss = self.evaluate_ipo_grp_loss(dataset)
                    val_grp_loss = self.evaluate_ipo_grp_loss(val_dataset)

                kl_dist=self.evaluate_KL(env=env,states=test_dataset)
                                
                #Evaluate the reward on the test dataset:
                #print(optimal_reward,self.evaluate_reward(env=env, 
                #                           states=test_dataset))
                rew_err = [float(a - b)/a for a, b in zip(optimal_reward,self.evaluate_reward(env=env, 
                                           states=test_dataset) )]
                formatted_rew=", ".join([f"{reward:.4f}" for reward in rew_err])
                
                formatted_kl=", ".join([f"{kld:.4f}" for kld in kl_dist])

                max_reward_err=max(rew_err)
                max_reward_err_index=rew_err.index(max_reward_err)

                max_kl_dist=max(kl_dist)
                max_kl_dist_index=kl_dist.index(max_kl_dist)

                max_train_grp_loss=np.max(train_grp_loss)
                max_val_grp_loss=np.max(val_grp_loss)
                max_cur_train_grp_loss=np.max(self.group_loss)
                max_train_grp_loss_index=np.argmax(train_grp_loss)
                max_val_grp_loss_index=np.argmax(val_grp_loss)
                max_cur_train_grp_loss_index=np.argmax(self.group_loss)


                logging_str = (f"Iteration: {step: d}, train_loss: {train_loss: .4f}, "
                            f"val_loss: {val_loss: .4f}, grad_norm: {grad_norm:.4f}, "
                            f"reward_err: {formatted_rew}, KL_dist: {formatted_kl}, param: {self.param}"
                            f"train_grp_loss: {train_grp_loss}, val_grp_loss: {val_grp_loss}, "
                            f"train_hist_grp_loss: {self.hist_group_loss}, cur_train_grp_loss: {self.group_loss},"
                             f"max_reward_err: {max_reward_err: .4f}, max_reward_err_index: {max_reward_err_index}, "
                            f"max_kl_dist: {max_kl_dist: .4f}, max_kl_dist_index: {max_kl_dist_index}, "
                            f"max_train_grp_loss: {max_train_grp_loss: .4f}, max_train_grp_loss_index: {max_train_grp_loss_index}, "
                            f"max_val_grp_loss: {max_val_grp_loss: .4f}, max_val_grp_loss_index: {max_val_grp_loss_index}, "
                            f"max_cur_train_grp_loss: {max_cur_train_grp_loss: .4f}, max_cur_train_grp_loss_index: {max_cur_train_grp_loss_index}, ")
                
                if self.wandb_use:
                    d_wandb = {
                        "Iteration": step, "train_loss": train_loss, 
                            "val_loss": val_loss, "grad_norm": grad_norm, 
                             "max_reward_err": max_reward_err , "max_reward_err_index": max_reward_err_index, 
                            "max_kl_dist" : max_kl_dist, "max_kl_dist_index": max_kl_dist_index, 
                            "max_train_grp_loss": max_train_grp_loss, "max_train_grp_loss_index": max_train_grp_loss_index, 
                            "max_val_grp_loss": max_val_grp_loss, "max_val_grp_loss_index": max_val_grp_loss_index, 
                            "max_cur_train_grp_loss": max_cur_train_grp_loss, "max_cur_train_grp_loss_index": max_cur_train_grp_loss_index
                    }
                    # Assuming rew_err is a list
                    for i, err in enumerate(rew_err):
                        key = f"reward_err_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                        d_wandb[key] = err
                    for i, param in enumerate(self.param):
                        key = f"reward_param_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                        d_wandb[key] = param 
                    for i, kld in enumerate(kl_dist):
                        key = f"KL_distance_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                        d_wandb[key] = kld
                    for i, hist_grp_ls in enumerate(self.hist_group_loss):
                        key = f"hist_group_loss_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                        d_wandb[key] = hist_grp_ls 
                    for i, grp_ls in enumerate(self.group_loss):
                        key = f"cur_group_loss_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                        d_wandb[key] = grp_ls 
                    for i, grp_ls in enumerate(train_grp_loss):
                        key = f"train_group_loss_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                        d_wandb[key] = grp_ls   
                    for i, grp_ls in enumerate(val_grp_loss):
                        key = f"val_group_loss_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                        d_wandb[key] = grp_ls          
                    wandb.log(d_wandb)
                
                if self.logger:
                    self.logger.info(logging_str)
                else:
                    print(logging_str)
        if self.ipo_grad_type=='justdpo':
            train_loss = self.evaluate_loss(dataset)
            val_loss = self.evaluate_loss(val_dataset)
        else:
            train_loss = self.evaluate_ipo_loss(dataset)
            val_loss = self.evaluate_ipo_loss(val_dataset)

        if self.ipo_grad_type=='justdpo':
            train_grp_loss = self.evaluate_grp_loss(dataset)
            val_grp_loss = self.evaluate_grp_loss(val_dataset)
        else:
            train_grp_loss = self.evaluate_ipo_grp_loss(dataset)
            val_grp_loss = self.evaluate_ipo_grp_loss(val_dataset)
        
        kl_dist=self.evaluate_KL(env=env,states=test_dataset)

        formatted_kl=", ".join([f"{kld:.4f}" for kld in kl_dist])

        #Evaluate the reward on the test dataset:
        #print(optimal_reward,self.evaluate_reward(env=env, 
        #                           states=test_dataset))
        rew_err = [float(a - b)/a for a, b in zip(optimal_reward,self.evaluate_reward(env=env, 
                                    states=test_dataset) )]
        formatted_rew=", ".join([f"{reward:.4f}" for reward in rew_err])

        max_reward_err=max(rew_err)
        max_reward_err_index=rew_err.index(max_reward_err)

        max_kl_dist=max(kl_dist)
        max_kl_dist_index=kl_dist.index(max_kl_dist)

        max_train_grp_loss=np.max(train_grp_loss)
        max_val_grp_loss=np.max(val_grp_loss)
        max_cur_train_grp_loss=np.max(self.group_loss)
        max_train_grp_loss_index=np.argmax(train_grp_loss)
        max_val_grp_loss_index=np.argmax(val_grp_loss)
        max_cur_train_grp_loss_index=np.argmax(self.group_loss)

        logging_str = (f"Iteration: {step: d}, train_loss: {train_loss: .4f}, "
                    f"val_loss: {val_loss: .4f}, grad_norm: {grad_norm:.4f}, "
                    f"reward_err: {formatted_rew}, KL_dist: {formatted_kl}, param: {self.param}"
                    f"train_grp_loss: {train_grp_loss}, val_grp_loss: {val_grp_loss}, "
                    f"train_hist_grp_loss: {self.hist_group_loss}, cur_train_grp_loss: {self.group_loss},"
                    f"max_reward_err: {max_reward_err: .4f}, max_reward_err_index: {max_reward_err_index}, "
                    f"max_kl_dist: {max_kl_dist: .4f}, max_kl_dist_index: {max_kl_dist_index}, "
                    f"max_train_grp_loss: {max_train_grp_loss: .4f}, max_train_grp_loss_index: {max_train_grp_loss_index}, "
                    f"max_val_grp_loss: {max_val_grp_loss: .4f}, max_val_grp_loss_index: {max_val_grp_loss_index}, "
                    f"max_cur_train_grp_loss: {max_cur_train_grp_loss: .4f}, max_cur_train_grp_loss_index: {max_cur_train_grp_loss_index}, ")
        
        if self.wandb_use:
            d_wandb = {
                "Iteration": step, "train_loss": train_loss, 
                    "val_loss": val_loss, "grad_norm": grad_norm, 
                    "max_kl_dist" : max_kl_dist, "max_kl_dist_index": max_kl_dist_index, 
                    "max_train_grp_loss": max_train_grp_loss, "max_train_grp_loss_index": max_train_grp_loss_index, 
                    "max_val_grp_loss": max_val_grp_loss, "max_val_grp_loss_index": max_val_grp_loss_index, 
                    "max_cur_train_grp_loss": max_cur_train_grp_loss, "max_cur_train_grp_loss_index": max_cur_train_grp_loss_index
            }
            # Assuming rew_err is a list
            for i, err in enumerate(rew_err):
                key = f"reward_err_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = err
            for i, param in enumerate(self.param):
                key = f"reward_param_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = param  
            for i, kld in enumerate(kl_dist):
                key = f"KL_distance_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = kld  
            for i, grp_ls in enumerate(train_grp_loss):
                key = f"train_group_loss_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = grp_ls  
            for i, hist_grp_ls in enumerate(self.hist_group_loss):
                key = f"hist_group_loss_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = hist_grp_ls 
            for i, grp_ls in enumerate(self.group_loss):
                key = f"cur_group_loss_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = grp_ls 
            for i, grp_ls in enumerate(val_grp_loss):
                key = f"val_group_loss_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = grp_ls    
            wandb.log(d_wandb)
        
        if self.logger:
            self.logger.info(logging_str)
        else:
            print(logging_str)
        rew = self.evaluate_reward(env, test_dataset)
        #rew = float(rew)
        return rew
    
    def train_by_cvxpy(self, dataset: List[GroupTransition], env: GroupLinearBandit) -> float:
        pref_features, non_pref_features = [], []
        pref_ref_policy, non_pref_ref_policy = [], []
        for transition in dataset:
            state, action_one, action_two, group_id, pref = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.group_id,
                transition.pref,
            )
            if pref == 1:
                pref_act = action_two
                non_pref_act = action_one
            else:
                pref_act = action_one
                non_pref_act = action_two

            feature_pref_act, feature_non_pref_act = (
                self.feature_func(state, pref_act, group_id),
                self.feature_func(state, non_pref_act, group_id),
            )
            pref_features.append(feature_pref_act)
            non_pref_features.append(feature_non_pref_act)

            act_prob = self.ref_policy(state)
            pref_ref_policy.append(act_prob[pref_act])
            non_pref_ref_policy.append(act_prob[non_pref_act])

        pref_features = np.stack(pref_features, axis=0)
        non_pref_features = np.stack(non_pref_features, axis=0)

        pref_ref_policy = np.stack(pref_ref_policy, axis=0)
        non_pref_ref_policy = np.stack(non_pref_ref_policy, axis=0)

        theta = cp.Variable(self.feature_dim)
        log_policy_diff = (non_pref_features - pref_features) @ theta
        log_ref_policy_diff = cp.log(non_pref_ref_policy) - cp.log(pref_ref_policy)

        tmp = self.reg_coef * (log_policy_diff - log_ref_policy_diff)

        loss = cp.sum(cp.logistic(tmp)) / len(dataset)
        problem = cp.Problem(cp.Minimize(loss))
        problem.solve(solver="ECOS", verbose=False)

        theta_arr = np.array(theta.value)

        self.param = theta_arr
        loss, reward = self.evaluate_loss(dataset), self.evaluate_reward(env)
        if self.logger:
            self.logger.info("Train by cvxopt.")
            self.logger.info(f"Loss calculated by cvxopt: {problem.value: .4f}.")
            self.logger.info(f"Loss: {loss: .4f}, reward: {reward: .4f}.")
        else:
            print("Train by cvxopt.")
            print(f"Loss calculated by cvxopt: {problem.value: .4f}.")
            print(f"Loss: {loss: .4f}, reward: {reward: .4f}.")

        return reward

    def evaluate_reward(self, env: GroupLinearBandit, states:Union[list, None] ) -> float:
        policy = self.ret_policy()
        rew = env.evaluate_reward_group_wise(policy,states)

        return rew
 
    @property
    def get_param(self) -> np.ndarray:
        return self.param
    
    def evaluate_KL(self, env: GroupLinearBandit, states:Union[list, None] ) -> float:
        policy = self.ret_policy()
        kl_dist = env.evaluate_KL_group_wise(policy,states)

        return kl_dist