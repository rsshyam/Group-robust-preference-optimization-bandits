import copy
import cvxpy as cp
import numpy as np
from typing import List,Set, Union
from envs.group_linear_bandit import GroupLinearBandit
from utils.collect_data import GroupTransition, ret_uniform_policy, collect_preference_data
from utils.utils import softmax, sigmoid
from utils.logger import Logger
import random
import wandb
from collections import defaultdict

class GroupRobustDirectPolicyOptimizationVectorised:
    def __init__(
        self,
        state_dim: int,                                 ## state s drawn as vector of `state_dim` elements from Uniform(0,1)
        action_num: int,                                ## number of actions in Action Space
        group_num: int,                                 ## number of groups
        feature_dim: int,                               ## feature_dim = 2 * state_dim (num elements in vector φ(s,a,g) )
        feature_func,                                   ## φ(s,a,g)
        ref_policy,                                     ## π_ref(a|s)
        reg_coef: float,                                ## β scaling in the DPO gradient & loss -- controls KL Divergence from π_ref
        step_size: float,                               ## η_θ step size for Gradient Descent on the DPO/IPO loss (if not is_adaptive)
        num_iters: int,                                 ## number of update steps on Training dataset
        exp_step_size: float,                           ## η_q step size for group weights
        batch_size: int,                                ## batch computation instead of for-loop over each datapoint in D_pref
        exp_adaptive: float = 0,                        ## adaptive exp_step_size
        is_adaptive: bool = False,                      ## if is_adaptive, step size in Update step is adaptive to the historical grad
        ada_coef: float = None,                         ## coef scaling the inverted-sqrt historical grad in Update step if is_adaptive
        logger: Logger = None,                          ## logger
        wandb_use: bool = False,                        ## recording results in WandB
        weighted_batches: bool = False,                 ## random sample of size G (num groups) among batched datapoints
        adj: str = None,                                ## adjustment by the generalisation error in the loss & grad calculation
        importance_sampling: bool = False,              ## importance sampling by non-trainable weight q for each group
        importance_sampling_weights: str = 'None',      ## non-trainable importance weights for each group
        ipo_grad_type: str = 'justdpo',                 ## `justdpo` (vectorised version), `linear` (IPO), or `log` (IPO)
        param_limit: int = 1,                           ## elements of vector θ range in [0, param_limit]
        use_closed_form: bool=False,                    ## closed-form regression solution for IPO
        lamba: float =0,                                ## L2 regularisation for closed-form regression of IPO objective in Linear Bandits case
        l2_reg_rdpo: float = 0,                         ## L2 regularisation for vectorised RDPO
        reg_by_group_weights: float = 0,                ## regularisation on vectorised RDPO by subtracting step*group_weights^2
        train_agent: bool=True,                         ## if True, use self.train(); else, use self.random_train() func
        report_iter: int = 2000                         ## log metrics after these iters
    ) -> None:
        self.state_dim = state_dim
        self.action_num = action_num
        self.feature_dim = feature_dim
        self.group_num = group_num
        self.feature_func = feature_func
        self.step_size = step_size
        self.num_iters = num_iters
        self.ref_policy = ref_policy
        self.reg_coef = reg_coef
        self.logger = logger
        self.wandb_use=wandb_use
        self.ipo_grad_type=ipo_grad_type

        self.group_weights=np.ones(group_num)/group_num
        self.exp_step_size=exp_step_size
        self.batch_size=batch_size

        self.exp_adaptive = exp_adaptive
        self.is_adaptive = is_adaptive
        self.ada_coef = ada_coef
        self.hist_grad_squared_norm = 0.0
        self.hist_group_loss=np.zeros(group_num)
        # initialize the policy parameter
        self.param = np.random.uniform(0, param_limit, self.feature_dim)
        print(f'PARAM INIT: {self.param}\n\n')

        self.weighted_batches=weighted_batches
        self.importance_sampling=importance_sampling
        if importance_sampling_weights=='None':
            self.importance_sampling_weights=None
        else:
            self.importance_sampling_weights=importance_sampling_weights

        if adj is not None:
            # process generalization adjustment stuff
            adjustments = [float(c) for c in adj.split(',')]
            assert len(adjustments) in (1, self.group_num)
            if len(adjustments)==1:
                adjustments = adjustments[0]*np.ones(self.group_num)
            else:
                adjustments = np.array(adjustments)
            self.adj = adjustments
        else:
            self.adj = np.zeros(self.group_num)
        
        self.use_closed_form=use_closed_form
        self.lamba=lamba
        self.l2_reg_rdpo = l2_reg_rdpo
        self.reg_by_group_weights = reg_by_group_weights
        self.train_agent=train_agent
        self.report_iter = report_iter
        print(self.step_size,weighted_batches,self.adj)
        
    def ret_action_prob(self, state: np.ndarray, group_id: int) -> np.ndarray:
        arr = np.zeros(self.action_num, np.float32)
        for action_idx in range(self.action_num):
            feature = self.feature_func(state, action_idx, group_id)
            arr[action_idx] = np.dot(feature, self.param)
        prob = softmax(arr)
        return prob

    def ret_policy(self):
        action_num = self.action_num
        feature_func = copy.deepcopy(self.feature_func)
        param = self.param

        def policy(state: np.ndarray, group_id: int) -> np.ndarray:
            #print('policy used -- PARAM ', param)
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

    def update_once(self, dataset: List[GroupTransition]) -> float:
        grad = np.zeros_like(self.param)
        feature_diff_all = np.zeros((len(dataset), self.feature_dim))
        pref_act_all = []
        non_pref_act_all = []
        
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

        log_ratio_diff = self.reg_coef * feature_diff_all @ self.param.reshape(self.feature_dim,1) # VECTORISATION
        coef = sigmoid(-log_ratio_diff)
        neg_cur_data_grad = (self.reg_coef * coef * feature_diff_all)
        grad = np.sum(-neg_cur_data_grad, axis=1) / len(dataset)

        self.hist_grad_squared_norm += np.sum(np.square(grad))
        if self.is_adaptive:
            step_size = self.ada_coef / np.sqrt(self.hist_grad_squared_norm)
        else:
            step_size = self.step_size
        self.param = self.param - step_size * grad
        return np.sqrt(np.sum(np.square(grad)))
    
    def batch_update_once_NONVEC(self, dataset: List[GroupTransition],batch_size: int, unique_group_ids: Set[int]) -> float:

        def sample_group_transition(group_id):
            #print(group_id)
            #print(dataset[0].group_id)
            group_transitions_with_id = [transition for transition in dataset if transition.group_id == group_id]
            return random.choice(group_transitions_with_id)

        if self.weighted_batches==True:
            if len(unique_group_ids)==self.group_num:
                
                group_id_mat=np.floor(np.random.uniform(
                    0, self.group_num, size=(batch_size)
                )).astype(int)#sample group_ids according to batchsize
                #print(group_id_mat)
            else:
                group_id_mat = np.array(random.choices(list(unique_group_ids), k=batch_size)).astype(int)
                #print(group_id_mat)
            # Sample GroupTransitions for each group_id in the array
            sampled_group_transitions = [sample_group_transition(group_id) for group_id in group_id_mat]#within that group choose a transition
        else:
            if batch_size<len(dataset):
                sampled_group_transitions=random.choices(dataset,k=batch_size)
            else:
                sampled_group_transitions=dataset

        # Display the sampled GroupTransitions
        #for group_id, sampled_transition in zip(group_id_mat, sampled_group_transitions):
        #    print(f"Group ID: {group_id}, Sampled Transition: {sampled_transition}")

        group_loss_tot=np.zeros(self.group_num)
        cur_group_counts=np.zeros(self.group_num)
        
        for transition in sampled_group_transitions:
            grad = np.zeros_like(self.param)
            group_loss=np.zeros(self.group_num)

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

            log_ratio_diff = self.reg_coef * (feat_pref_act - feat_non_pref_act) @ self.param

            if self.ipo_grad_type=='linear':
                lin_diff=(feat_pref_act-feat_non_pref_act)@(self.param)-0.5*(1/self.reg_coef)
                coef=-2*lin_diff/self.reg_coef
            elif self.ipo_grad_type=='log':
                log_diff=(
                    np.log((cur_policy_act_prob[pref_act]*ref_policy_act_prob[non_pref_act])/(cur_policy_act_prob[non_pref_act]*ref_policy_act_prob[pref_act])+1e-6 )
                )
                coef=-2*(log_diff-0.5*(1/self.reg_coef))/self.reg_coef
            elif self.ipo_grad_type=='justdpo':
                coef = sigmoid(-log_ratio_diff)
            else:
                raise ValueError('value not implemented')
            
            neg_cur_data_grad = (
                self.reg_coef * coef * (feat_pref_act - feat_non_pref_act)
            )
            grad = - self.group_weights[group_id]*neg_cur_data_grad#weighted gradient calculation

            if self.ipo_grad_type=='linear':
                lin_diff=(feat_pref_act-feat_non_pref_act)@(self.param)-0.5*(1/self.reg_coef)
                group_loss[group_id]+=np.square(lin_diff)+self.adj[group_id]/np.sqrt(self.group_counts[group_id])
            elif self.ipo_grad_type=='log':
                #print('log')
                log_diff=(
                    np.log((cur_policy_act_prob[pref_act]*ref_policy_act_prob[non_pref_act])/(cur_policy_act_prob[non_pref_act]*ref_policy_act_prob[pref_act])+1e-6 )
                )
                group_loss[group_id]+=np.square((log_diff-0.5*(1/self.reg_coef)))+self.adj[group_id]/np.sqrt(self.group_counts[group_id])
            elif self.ipo_grad_type=='justdpo':
                group_loss[group_id]=-np.log(sigmoid(log_ratio_diff))+self.adj[group_id]/np.sqrt(self.group_counts[group_id]) #calculate group losses
            else:
                raise ValueError('value not implemented')

            #grad /= len(sampled_group_transitions)
            #group_loss=group_loss/cur_group_counts
            if self.importance_sampling==False:
                #print(self.group_weights,group_loss,np.exp(self.exp_step_size*group_loss))
                self.group_weights[group_id]=self.group_weights[group_id]*np.exp(self.exp_step_size*group_loss[group_id])#update weights based on group loss calculated
                #print(self.group_weights)
                self.group_weights=self.group_weights/np.sum(self.group_weights)#normalize the weights
            self.hist_grad_squared_norm += np.sum(np.square(grad))
            self.hist_group_loss[group_id]+=group_loss[group_id]
            group_loss_tot += group_loss[group_id]
            cur_group_counts[group_id] += 1

            if self.is_adaptive:
                step_size = self.ada_coef / np.sqrt(self.hist_grad_squared_norm)
            else:
                step_size = self.step_size
            self.param = self.param - step_size * (grad) 

        self.group_loss = group_loss_tot / cur_group_counts

        live_grad = 0.0
        return np.sqrt(np.sum(np.square(grad))), live_grad
    
    def batch_update_once(self, dataset: List[GroupTransition],batch_size: int, unique_group_ids: Set[int]) -> float:

        def sample_group_transition(group_id):
            #print(group_id)
            #print(dataset[0].group_id)
            group_transitions_with_id = [transition for transition in dataset if transition.group_id == group_id]
            return random.choice(group_transitions_with_id)

        if self.weighted_batches==True:
            if len(unique_group_ids)==self.group_num:
                
                group_id_mat=np.floor(np.random.uniform(
                    0, self.group_num, size=(batch_size)
                )).astype(int)#sample group_ids according to batchsize
                #print(group_id_mat)
            else:
                group_id_mat = np.array(random.choices(list(unique_group_ids), k=batch_size)).astype(int)
                #print(group_id_mat)
            # Sample GroupTransitions for each group_id in the array
            sampled_group_transitions = [sample_group_transition(group_id) for group_id in group_id_mat]#within that group choose a transition        ######################
        else:
            if batch_size<len(dataset):
                sampled_group_transitions=random.choices(dataset,k=batch_size)
            else:
                sampled_group_transitions=dataset

        grad = np.zeros_like(self.param)
        group_grad = np.zeros((self.group_num, self.param.shape[-1]))
        group_loss = np.zeros(self.group_num)
        cur_group_counts = np.zeros(self.group_num)

        group_id_idx_all = defaultdict(list)
        feature_diff_all = np.zeros((len(sampled_group_transitions), self.feature_dim))
        pref_act_all = []
        non_pref_act_all = []
        cur_policy_act_prob_all = np.zeros((len(sampled_group_transitions), self.action_num))
        ref_policy_act_prob_all = np.zeros((len(sampled_group_transitions), self.action_num))
        
        for idx, transition in enumerate(sampled_group_transitions):
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
                self.feature_func(state, non_pref_act, group_id),
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
                group_loss[group_id] = np.sum(-np.log(sigmoid(log_ratio_diff_all[group_indices])))+self.adj[group_id]/np.sqrt(self.group_counts[group_id]) #calculate group losses
        elif self.ipo_grad_type=='linear':
            lin_diff = feature_diff_all @ self.param.reshape(self.feature_dim,1) - 0.5*(1/self.reg_coef)
            coef = -2*lin_diff/self.reg_coef
            for group_id in range(self.group_num):
                group_indices = group_id_idx_all[group_id]
                group_loss[group_id] = np.sum(np.square(lin_diff[group_indices]))+self.adj[group_id]/np.sqrt(self.group_counts[group_id])
        elif self.ipo_grad_type=='log':
            row_indices = np.arange(cur_policy_act_prob_all.shape[0])
            log_diff = (
                np.log((cur_policy_act_prob_all[row_indices,pref_act_all]*ref_policy_act_prob_all[row_indices,non_pref_act])/
                       (cur_policy_act_prob_all[row_indices,non_pref_act]*ref_policy_act_prob_all[row_indices,pref_act_all])+1e-6 )
            )
            coef = -2*(log_diff-0.5*(1/self.reg_coef))/self.reg_coef
            for group_id in range(self.group_num):
                group_indices = group_id_idx_all[group_id]
                group_loss[group_id] = np.sum(np.square((log_diff[group_indices]-0.5*(1/self.reg_coef))))+self.adj[group_id]/np.sqrt(self.group_counts[group_id])
        else:
            raise ValueError('value not implemented')
        
        neg_cur_data_grad = (self.reg_coef * coef * feature_diff_all)
        grad = np.sum(-neg_cur_data_grad, axis=0) / len(sampled_group_transitions)               ############### had self.group_weights[group_id] scaling before
        group_loss /= cur_group_counts

        if self.l2_reg_rdpo != 0:
            grad += 2 * self.l2_reg_rdpo * self.param #/ len(sampled_group_transitions) # theta-gradient on loss L2 norm λ . ||θ||_F
        self.hist_grad_squared_norm += np.sum(np.square(grad))

        if self.importance_sampling is False:
            if self.exp_adaptive > 0:
                # Geometric exponential decay == self.exp_step_size /= self.exp_adaptive
                exp_step_size = self.exp_step_size * self.exp_adaptive / np.sqrt(self.hist_grad_squared_norm)
            else:
                exp_step_size = self.exp_step_size

            self.group_weights = self.group_weights*np.exp(exp_step_size*group_loss)#update weights based on group loss calculated
            self.group_weights = self.group_weights/np.sum(self.group_weights)#normalize the weights

        weighted_group_grad = np.zeros_like(group_grad)
        for group_id in range(self.group_num):
            group_indices = group_id_idx_all[group_id]
            group_grad_weighted_sum = np.sum(-neg_cur_data_grad[group_indices], axis=0) * self.group_weights[group_id]
            if self.importance_sampling==False: # divide grads by group counts in RDPO
                weighted_group_grad[group_id] = group_grad_weighted_sum / cur_group_counts[group_id] # len(sampled_group_transitions)  ############### had self.group_weights[group_id] scaling before
            else: # Importance Sampling has weights assigned to inv-freq and then group-grads divided by batch size
                weighted_group_grad[group_id] = group_grad_weighted_sum / len(sampled_group_transitions)
        
        if self.l2_reg_rdpo != 0:
            for g in range(self.group_num):
                group_loss[g] += self.l2_reg_rdpo * np.square(np.linalg.norm(self.param)) #/ cur_group_counts # regularisation L2
                weighted_group_grad[g] += 2 * self.l2_reg_rdpo * self.param #* self.group_weights[g] #/ cur_group_counts[g]
        elif self.reg_by_group_weights != 0:
            group_loss -= self.reg_by_group_weights * np.square(self.group_weights) # Paper Theorem 3.1
            # grad is invariant to this negative term, so no update

        if self.is_adaptive:
            step_size = self.ada_coef / np.sqrt(self.hist_grad_squared_norm)
        else:
            step_size = self.step_size

        self.hist_group_loss += group_loss
        self.group_loss = group_loss

        #self.param = self.param - step_size * (grad) ## WRONG....
        self.param = self.param - step_size * np.sum(weighted_group_grad, axis=0) #(self.group_weights @ group_grad) # not just grad but group-grad scaled
        #print(f'GROUP GRAD: grp0 {group_grad[0]} -- grp1 {group_grad[1]} -- total {group_grad[0] + group_grad[1]} -- matmul {2*(self.group_weights @ group_grad)} || PARAM {self.param}\n')

        #print(f'PARAM -- NO REG {param_noreg} -- REG {self.param}')
        live_grad = 0.0

        self.theta_update = step_size * np.sum(weighted_group_grad, axis=0)
        self.total_loss = sum(self.group_loss)

        return np.sqrt(np.sum(np.square(grad))), live_grad
    
    def batch_update_once_GROUPBATCH(self, dataset: List[GroupTransition],batch_size: int, unique_group_ids: Set[int]) -> float:
        """ Update with sample over entire single group only """

        def sample_batch_from_group(group_id):
            """ Sample a batch of size `batch_size` without replacement, all from the same group """
            group_transitions_with_id = [transition for transition in dataset if transition.group_id == group_id]
            if batch_size >= len(group_transitions_with_id):
                return group_transitions_with_id
            random.shuffle(group_transitions_with_id) # in-place random permutation
            return group_transitions_with_id[:batch_size] # random non-replacement sample from same group

        if self.weighted_batches==True:
            raise NotImplementedError
        else:
            rand_group = random.choice(list(unique_group_ids)) # choose a group at random
            sampled_group_transitions=sample_batch_from_group(rand_group) # sample a batch of transitions from this group

        # single grad & loss placeholder as all samples are from same group
        grad = np.zeros_like(self.param)
        loss = 0.0
        
        for transition in sampled_group_transitions:
            state, action_one, action_two, group_id, pref = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.group_id,
                transition.pref,
            )
            assert(group_id == rand_group), "Wrong group found in `batch_update_once` iteration."

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
                log_ratio_diff = self.reg_coef * (feat_pref_act - feat_non_pref_act) @ (self.param) # VECTORISED
                coef = sigmoid(-log_ratio_diff)
                loss += -np.log(sigmoid(log_ratio_diff))+self.adj[group_id]/np.sqrt(self.group_counts[group_id]) #calculate group losses
            elif self.ipo_grad_type=='linear':
                lin_diff = (feat_pref_act-feat_non_pref_act)@(self.param)-0.5*(1/self.reg_coef)
                coef = -2*lin_diff/self.reg_coef
                loss += np.square(lin_diff)+self.adj[group_id]/np.sqrt(self.group_counts[group_id])
            elif self.ipo_grad_type=='log':
                log_diff = (
                    np.log((cur_policy_act_prob[pref_act]*ref_policy_act_prob[non_pref_act])/(cur_policy_act_prob[non_pref_act]*ref_policy_act_prob[pref_act])+1e-6 )
                )
                coef = -2*(log_diff-0.5*(1/self.reg_coef))/self.reg_coef
                loss += np.square((log_diff-0.5*(1/self.reg_coef)))+self.adj[group_id]/np.sqrt(self.group_counts[group_id])
            else:
                raise ValueError('value not implemented')
            
            neg_cur_data_grad = (
                self.reg_coef * coef * (feat_pref_act - feat_non_pref_act)
            )
            grad -= neg_cur_data_grad
            #grad -= 2*self.group_weights[group_id]*neg_cur_data_grad ## WRONG... why 2 factor? why gather all grads from diff groups?
        
        # averaged grad & loss for this group batch
        grad /= len(sampled_group_transitions)
        loss /= len(sampled_group_transitions)

        # update the weights for the randomly-selected group in this batch (& renormalise weights across all groups)
        if self.importance_sampling==False:
            # WEIGHTS OF BATCH GROUP
            self.group_weights[rand_group] = self.group_weights[rand_group]*np.exp(self.exp_step_size*loss)#update weights based on group loss calculated
            # RE-NORMALISATION ACROSS ALL GROUPS
            self.group_weights = self.group_weights/np.sum(self.group_weights)#normalize the weights
        
        # Update gradient
        updated_grad = grad * self.group_weights[rand_group]
        
        self.hist_grad_squared_norm += np.sum(np.square(updated_grad))
        self.hist_group_loss += loss
        
        self.group_loss = np.zeros(self.group_num)
        self.group_loss[rand_group] = loss

        if self.is_adaptive:
            step_size = self.ada_coef / np.sqrt(self.hist_grad_squared_norm)
        else:
            step_size = self.step_size

        #self.param = self.param - step_size * (grad) ## WRONG....
        self.param = self.param - step_size * updated_grad # not just grad but group-grad scaled

        live_grad = 0.0
        return np.sqrt(np.sum(np.square(updated_grad))), live_grad
    
    def batch_update_closed_form(self, dataset: List[GroupTransition],batch_size: int, unique_group_ids: Set[int]) -> float:

        def sample_group_transition(group_id):
            #print(group_id)
            #print(dataset[0].group_id)
            group_transitions_with_id = [transition for transition in dataset if transition.group_id == group_id]
            return random.choice(group_transitions_with_id)

        if self.weighted_batches==True:
            if len(unique_group_ids)==self.group_num:
                
                group_id_mat=np.floor(np.random.uniform(
                    0, self.group_num, size=(batch_size)
                )).astype(int)#sample group_ids according to batchsize
                #print(group_id_mat)
            else:
                group_id_mat = np.array(random.choices(list(unique_group_ids), k=batch_size)).astype(int)
                #print(group_id_mat)
            # Sample GroupTransitions for each group_id in the array
            sampled_group_transitions = [sample_group_transition(group_id) for group_id in group_id_mat]#within that group choose a transition
        else:
            if batch_size<len(dataset):
                sampled_group_transitions=random.choices(dataset,k=batch_size)
            else:
                sampled_group_transitions=dataset
            #print(len(sampled_group_transitions),'non_weighted')


        grad = np.zeros_like(self.param)
        group_grad = np.zeros((self.group_num, self.param.shape[-1]))
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
                group_loss[group_id] = np.sum(-np.log(sigmoid(log_ratio_diff_all[group_indices]))+self.adj[group_id]/np.sqrt(self.group_counts[group_id])) #calculate group losses
        elif self.ipo_grad_type=='linear':
            lin_diff = feature_diff_all @ self.param.reshape(self.feature_dim,1) - 0.5*(1/self.reg_coef)
            coef = -2*lin_diff/self.reg_coef
            for group_id in range(self.group_num):
                group_indices = group_id_idx_all[group_id]
                group_loss[group_id] = np.sum(np.square(lin_diff[group_indices])+self.adj[group_id]/np.sqrt(self.group_counts[group_id]))
        elif self.ipo_grad_type=='log':
            row_indices = np.arange(cur_policy_act_prob_all.shape[0])
            log_diff = (
                np.log((cur_policy_act_prob_all[row_indices,pref_act_all]*ref_policy_act_prob_all[row_indices,non_pref_act])/
                       (cur_policy_act_prob_all[row_indices,non_pref_act]*ref_policy_act_prob_all[row_indices,pref_act_all])+1e-6 )
            )
            coef = -2*(log_diff-0.5*(1/self.reg_coef))/self.reg_coef
            for group_id in range(self.group_num):
                group_indices = group_id_idx_all[group_id]
                group_loss[group_id] = np.sum(np.square((log_diff[group_indices]-0.5*(1/self.reg_coef)))+self.adj[group_id]/np.sqrt(self.group_counts[group_id]))
        else:
            raise ValueError('value not implemented')
        
        neg_cur_data_grad = (self.reg_coef * coef * feature_diff_all)
        for group_id in range(self.group_num):
            group_indices = group_id_idx_all[group_id]
            group_grad[group_id] = np.sum(-self.group_weights[group_id]*neg_cur_data_grad[group_indices], axis=0) / len(sampled_group_transitions)   ############### had self.group_weights[group_id] scaling before
        grad = np.sum(-self.group_weights[group_id]*neg_cur_data_grad, axis=0) / len(sampled_group_transitions)     ############### had self.group_weights[group_id] scaling before
        group_loss /= cur_group_counts
            
        for group_id in range(self.group_num):
            group_grad[group_id] /= cur_group_counts[group_id]
        group_loss=group_loss/cur_group_counts

        if self.importance_sampling==False:
            #print(self.group_weights,group_loss,np.exp(self.exp_step_size*group_loss))
            self.group_weights=self.group_weights*np.exp(self.exp_step_size*group_loss)#update weights based on group loss calculated
            #print(self.group_weights)
            self.group_weights=self.group_weights/np.sum(self.group_weights)#normalize the weights
        self.hist_grad_squared_norm += np.sum(np.square(grad))
        self.hist_group_loss+=group_loss
        self.group_loss=group_loss
        if self.is_adaptive:
            step_size = self.ada_coef / np.sqrt(self.hist_grad_squared_norm)
        else:
            step_size = self.step_size
        #print(grad)
        live_grad=self.WeightedRegression(sampled_group_transitions,self.lamba)
        #self.param=np.array([1.0,2.0])
        return np.sqrt(np.sum(np.square(grad))),live_grad
    
    def WeightedRegression(self, dataset: List[GroupTransition], lamba: float)-> float:
        print('WEIGHTED REGRESSION\n\n')
        Y=[]
        w=[]
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
            w.append(self.group_weights[group_id])
        Y=np.array(Y)
        w=np.array(w)
        #print(w)
        #print(Y.shape,np.diag(w).shape,(Y@self.param).T.shape,((Y@self.param).T-1/(2*self.reg_coef)).dot(Y).shape)
        coef=np.linalg.inv(Y.transpose()@np.diag(w)@Y+lamba*np.eye(Y.shape[1]))
        #print(np.linalg.det(np.matmul(Y.transpose(),Y)))
        variate=np.matmul(np.matmul(Y.transpose(),np.diag(w)),np.ones([len(dataset),1]))
        self.param=np.matmul(coef,variate).ravel()/(2*self.reg_coef)
        live_grad=(np.diag(w).dot((Y@self.param).T-1/(2*self.reg_coef))).dot(Y)+lamba*self.param
        return np.sqrt(np.sum(np.square(live_grad)))
    
    def evaluate_ipo_loss(self, dataset: List[GroupTransition], policy=None) -> float:
        """
        Evaluate the loss on the dataset for any policy.
        """
        if policy is None:
            policy = self.ret_policy()

        loss = 0.0
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
    
    def evaluate_weighted_ipo_loss(self, dataset: List[GroupTransition], policy=None) -> float:
        """
        Evaluate the loss on the dataset for any policy.
        """
        if policy is None:
            policy = self.ret_policy()

        loss = 0.0
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

        loss = np.sum(self.group_weights*np.square(coef)+self.adj[group_id]/np.sqrt(self.group_counts[group_id])) / len(dataset)
        loss = loss*self.group_num###for correct comparison as unweighted train loss should multiply 1/num_groups to all
        return loss
    
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
            loss[group_id] = np.sum(np.square(coef[group_indices]) + self.adj[group_id]/np.sqrt(self.group_counts[group_id]))

        loss = loss/counts
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
        loss /= len(dataset)

        if self.l2_reg_rdpo != 0:
            loss += self.l2_reg_rdpo * np.square(np.linalg.norm(self.param)) #/ len(dataset) # regularisation L2

        return loss
    
    def evaluate_weighted_loss(self, dataset: List[GroupTransition], policy=None) -> float:
        """
        Evaluate the loss on the dataset for any policy.
        """
        if policy is None:
            policy = self.ret_policy()

        loss = 0.0

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
            group_id_idx_all[group_id].append(idx)

        log_ratio_diff = self.reg_coef * feature_diff_all @ self.param.reshape(self.feature_dim,1)
        for group_id in range(self.group_num):
            group_indices = group_id_idx_all[group_id]
            loss += np.sum(-self.group_weights[group_id]*np.log(sigmoid(log_ratio_diff[group_indices]))+self.adj[group_id]/np.sqrt(self.group_counts[group_id]))
        loss /= len(dataset)

        if self.l2_reg_rdpo != 0:
            loss += self.l2_reg_rdpo * np.square(np.linalg.norm(self.param)) #/ len(dataset) # regularisation L2

        loss=loss*self.group_num###for correct comparison as unweighted train loss should multiply 1/num_groups to all
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
        
        loss = loss/ counts # len(dataset) #

        if self.l2_reg_rdpo != 0:
            loss += self.l2_reg_rdpo * np.square(np.linalg.norm(self.param)) #/ counts # regularisation L2
        elif self.reg_by_group_weights != 0:
            loss -= self.reg_by_group_weights * np.square(self.group_weights) # Paper Theorem 3.1
            # grad is invariant to this negative term, so no update

        return loss
    

    
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
        print(self.ipo_grad_type)
        #print(dataset)
        ratio=int(len(dataset)/self.batch_size)
        # Collect unique group IDs using set comprehension 
        unique_group_ids = {transition.group_id for transition in dataset}
        ##count group numbers

        group_counts = defaultdict(int)

        # Iterate through the transitions and count occurrences
        for group_transition in dataset:
            print(group_transition)
            group_counts[group_transition.group_id] += 1

        # Sort the dictionary items by group_id
        sorted_group_counts = sorted(group_counts.items(), key=lambda x: x[0])

        # Extract the counts and convert them to a NumPy array
        self.group_counts = np.array([count for _, count in sorted_group_counts])
        # Convert the dictionary to a NumPy array
        print(self.group_counts)
        if self.importance_sampling==True:
            self.weighted_batches=False
            if self.importance_sampling_weights:
                 # process generalization adjustment stuff
                imp_weights = [float(c) for c in self.importance_sampling_weights.split(',')]
                assert len(imp_weights) == self.group_num
                self.group_weights = np.array(imp_weights)
                self.group_weights=self.group_weights/np.sum(self.group_weights)
            else:
                self.group_weights=np.array([1/count for count in self.group_counts])
                self.group_weights=self.group_weights/np.sum(self.group_weights)
        self.logger.info(f'unique_group_ids: {unique_group_ids}')



        
        """
        step=-1
        
        if self.ipo_grad_type=='justdpo':
            train_loss = self.evaluate_loss(dataset)
            val_loss = self.evaluate_loss(val_dataset)
            train_wt_loss = self.evaluate_weighted_loss(dataset)
            val_wt_loss = self.evaluate_weighted_loss(val_dataset)
            train_grp_loss = self.evaluate_grp_loss(dataset)
            val_grp_loss = self.evaluate_grp_loss(val_dataset)
            grad_norm=self.evaluate_grad(dataset)
            live_grad=grad_norm
        else:
            train_loss = self.evaluate_ipo_loss(dataset)
            val_loss = self.evaluate_ipo_loss(val_dataset)
            train_wt_loss = self.evaluate_weighted_ipo_loss(dataset)
            val_wt_loss = self.evaluate_weighted_ipo_loss(val_dataset)
            train_grp_loss = self.evaluate_ipo_grp_loss(dataset)
            val_grp_loss = self.evaluate_ipo_grp_loss(val_dataset)
            grad_norm=self.evaluate_ipo_grad(dataset)
            live_grad=grad_norm
           
        self.group_loss=train_grp_loss
        kl_dist=self.evaluate_KL(env=env,states=test_dataset)
                        
        #Evaluate the reward on the test dataset:
        #rew = self.evaluate_reward(env=env, 
        #                           states=test_dataset)
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
                    f"val_loss: {val_loss: .4f}, grad_norm: {grad_norm:.4f}, live_grad: {live_grad:.4f}, "
                    f"reward_err: {formatted_rew}, KL_dist: {formatted_kl}, param: {self.param}, weights: {self.group_weights}, "
                    f"train_wt_loss: {train_wt_loss: .4f}, val_wt_loss: {val_wt_loss:.4f}, "
                    f"train_grp_loss: {train_grp_loss}, val_grp_loss: {val_grp_loss}, "
                    f"train_hist_grp_loss: {self.hist_group_loss}, cur_train_grp_loss: {self.group_loss}, "
                    f"max_reward_err: {max_reward_err: .4f}, max_reward_err_index: {max_reward_err_index}, "
                    f"max_kl_dist: {max_kl_dist: .4f}, max_kl_dist_index: {max_kl_dist_index}, "
                    f"max_train_grp_loss: {max_train_grp_loss: .4f}, max_train_grp_loss_index: {max_train_grp_loss_index}, "
                    f"max_val_grp_loss: {max_val_grp_loss: .4f}, max_val_grp_loss_index: {max_val_grp_loss_index}, "
                    f"max_cur_train_grp_loss: {max_cur_train_grp_loss: .4f}, max_cur_train_grp_loss_index: {max_cur_train_grp_loss_index}, ")
        if self.wandb_use:
            d_wandb = {
                "Iteration": step, "train_loss": train_loss, 
                    "val_loss": val_loss, "grad_norm": grad_norm, "live_grad": live_grad, 
                    "train_weighted_loss": train_wt_loss, "val_weighted_loss": val_wt_loss,
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
            for i, kld in enumerate(kl_dist):
                key = f"KL_distance_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = kld
            for i, param in enumerate(self.param):
                key = f"reward_param_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = param      
            for i, grp_wt in enumerate(self.group_weights):
                key = f"group_weight_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = grp_wt 
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

        """







        
        for step in range(ratio*self.num_iters):
            if self.use_closed_form:
                grad_norm,live_grad = self.batch_update_closed_form(dataset,self.batch_size,unique_group_ids)
            else:
                grad_norm,live_grad = self.batch_update_once(dataset,self.batch_size,unique_group_ids)
            if step % self.report_iter == 0:
                print('UPDATE PARAM GroupRobustDPO: ', self.theta_update)
                print('TOTAL LOSS GroupRobustDPO: ', self.total_loss)
                
                if self.ipo_grad_type=='justdpo':
                    train_loss = self.evaluate_loss(dataset)
                    val_loss = self.evaluate_loss(val_dataset)
                else:
                    train_loss = self.evaluate_ipo_loss(dataset)
                    val_loss = self.evaluate_ipo_loss(val_dataset)
                
                if self.ipo_grad_type=='justdpo':
                    train_wt_loss = self.evaluate_weighted_loss(dataset)
                    val_wt_loss = self.evaluate_weighted_loss(val_dataset)
                else:
                    train_wt_loss = self.evaluate_weighted_ipo_loss(dataset)
                    val_wt_loss = self.evaluate_weighted_ipo_loss(val_dataset)

                if self.ipo_grad_type=='justdpo':
                    train_grp_loss = self.evaluate_grp_loss(dataset)
                    val_grp_loss = self.evaluate_grp_loss(val_dataset)
                else:
                    train_grp_loss = self.evaluate_ipo_grp_loss(dataset)
                    val_grp_loss = self.evaluate_ipo_grp_loss(val_dataset)

                kl_dist=self.evaluate_KL(env=env,states=test_dataset)
                                
                #Evaluate the reward on the test dataset:
                #rew = self.evaluate_reward(env=env, 
                #                           states=test_dataset)
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
                            f"val_loss: {val_loss: .4f}, grad_norm: {grad_norm:.4f}, live_grad: {live_grad:.4f}, "
                            f"reward_err: {formatted_rew}, KL_dist: {formatted_kl}, param: {self.param}, weights: {self.group_weights}, "
                            f"train_wt_loss: {train_wt_loss: .4f}, val_wt_loss: {val_wt_loss:.4f}, "
                            f"train_grp_loss: {train_grp_loss}, val_grp_loss: {val_grp_loss}, "
                            f"train_hist_grp_loss: {self.hist_group_loss}, cur_train_grp_loss: {self.group_loss}, "
                            f"max_reward_err: {max_reward_err: .4f}, max_reward_err_index: {max_reward_err_index}, "
                            f"max_kl_dist: {max_kl_dist: .4f}, max_kl_dist_index: {max_kl_dist_index}, "
                            f"max_train_grp_loss: {max_train_grp_loss: .4f}, max_train_grp_loss_index: {max_train_grp_loss_index}, "
                            f"max_val_grp_loss: {max_val_grp_loss: .4f}, max_val_grp_loss_index: {max_val_grp_loss_index}, "
                            f"max_cur_train_grp_loss: {max_cur_train_grp_loss: .4f}, max_cur_train_grp_loss_index: {max_cur_train_grp_loss_index}, ")
                if self.wandb_use:
                    d_wandb = {
                        "Iteration": step, "train_loss": train_loss, 
                            "val_loss": val_loss, "grad_norm": grad_norm, "live_grad": live_grad, 
                            "train_weighted_loss": train_wt_loss, "val_weighted_loss": val_wt_loss,
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
                    for i, kld in enumerate(kl_dist):
                        key = f"KL_distance_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                        d_wandb[key] = kld
                    for i, param in enumerate(self.param):
                        key = f"reward_param_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                        d_wandb[key] = param      
                    for i, grp_wt in enumerate(self.group_weights):
                        key = f"group_weight_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                        d_wandb[key] = grp_wt 
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
            train_wt_loss = self.evaluate_weighted_loss(dataset)
            val_wt_loss = self.evaluate_weighted_loss(val_dataset)
        else:
            train_wt_loss = self.evaluate_weighted_ipo_loss(dataset)
            val_wt_loss = self.evaluate_weighted_ipo_loss(val_dataset)

        if self.ipo_grad_type=='justdpo':
            train_grp_loss = self.evaluate_grp_loss(dataset)
            val_grp_loss = self.evaluate_grp_loss(val_dataset)
        else:
            train_grp_loss = self.evaluate_ipo_grp_loss(dataset)
            val_grp_loss = self.evaluate_ipo_grp_loss(val_dataset)
        

        kl_dist=self.evaluate_KL(env=env,states=test_dataset)
                        
        formatted_kl=", ".join([f"{kld:.4f}" for kld in kl_dist])

        #Evaluate the reward on the test dataset:
        #rew = self.evaluate_reward(env=env, 
        #                           states=test_dataset)
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
                        f"val_loss: {val_loss: .4f}, grad_norm: {grad_norm:.4f},  live_grad: {live_grad:.4f}, "
                        f"reward_err: {formatted_rew}, KL_dist: {formatted_kl}, param: {self.param}, weights: {self.group_weights}, "
                        f"train_wt_loss: {train_wt_loss: .4f}, val_wt_loss: {val_wt_loss:.4f}, "
                        f"train_grp_loss: {train_grp_loss}, val_grp_loss: {val_grp_loss}, "
                        f"train_hist_grp_loss: {self.hist_group_loss}, cur_train_grp_loss: {self.group_loss}, "
                        f"max_reward_err: {max_reward_err: .4f}, max_reward_err_index: {max_reward_err_index}, "
                        f"max_kl_dist: {max_kl_dist: .4f}, max_kl_dist_index: {max_kl_dist_index}, "
                        f"max_train_grp_loss: {max_train_grp_loss: .4f}, max_train_grp_loss_index: {max_train_grp_loss_index}, "
                        f"max_val_grp_loss: {max_val_grp_loss: .4f}, max_val_grp_loss_index: {max_val_grp_loss_index}, "
                        f"max_cur_train_grp_loss: {max_cur_train_grp_loss: .4f}, max_cur_train_grp_loss_index: {max_cur_train_grp_loss_index}, ")
        if self.wandb_use:
            d_wandb = {
                "Iteration": step, "train_loss": train_loss, 
                    "val_loss": val_loss, "grad_norm": grad_norm,  "live_grad": live_grad,
                    "train_weighted_loss": train_wt_loss, "val_weighted_loss": val_wt_loss,
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
            for i, kld in enumerate(kl_dist):
                key = f"KL_distance_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = kld
            for i, param in enumerate(self.param):
                key = f"reward_param_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = param      
            for i, grp_wt in enumerate(self.group_weights):
                key = f"group_weight_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = grp_wt 
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
    
    def evaluate_KL(self, env: GroupLinearBandit, states:Union[list, None] ) -> float:
        policy = self.ret_policy()
        kl_dist = env.evaluate_KL_group_wise(policy,states)

        return kl_dist
 
    @property
    def get_param(self) -> np.ndarray:
        return self.param



