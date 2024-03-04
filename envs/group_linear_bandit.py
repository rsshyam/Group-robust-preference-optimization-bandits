import numpy as np
from scipy.stats import kendalltau, entropy
from typing import Union
from utils.collect_data import GroupTransition, process_pref_data, process_pref_grp_data
from scipy.special import softmax


class GroupLinearBandit:
    def __init__(
        self,
        state_dim: int,
        action_num: int,
        group_num: int,
        reward_param: np.ndarray,
        feature_func,
        num_trials_for_eval: int = None
    ) -> None:
        self.state_dim = state_dim
        self.action_num = action_num
        self.group_num = group_num
        self.action_space = [action_idx for action_idx in range(action_num)]
        self.group_ids=[group_id for group_id in range(group_num)]
        self.reward_param = reward_param
        self.feature_func = feature_func
        self.cur_state = np.random.uniform(0, 1, self.state_dim)

        self.num_trials_for_eval = num_trials_for_eval

    def reset(self) -> np.ndarray:
        self.cur_state = np.random.uniform(0, 1, self.state_dim)
        return self.cur_state

    def sample(self, action,group_id) -> float:
        assert action in self.action_space, "The input action is invalid."
        feature = self.feature_func(self.cur_state, action, group_id)
        print(feature,self.reward_param)
        assert np.shape(feature) == np.shape(
            self.reward_param
        ), "The feature is invalid."
        rew = np.dot(feature, self.reward_param)
        return rew

    def get_opt_policy(self):
        def opt_policy(state: np.ndarray, group_id: int):
            # compute the optimal policy by enumerating the action space
            feature_mat = np.array(
                [
                    self.feature_func(state, action_idx, group_id)
                    for action_idx in range(self.action_num)
                ],
                dtype=np.float32,
            )
            assert np.shape(feature_mat) == (
                self.action_num,
                self.reward_param.size,
            ), "The feature matrix is invalid."
            rew_vec = np.matmul(feature_mat, self.reward_param)
            optimal_action = np.argmax(rew_vec)
            action_prob = np.zeros(self.action_num, np.float32)
            action_prob[optimal_action] = 1.0

            return action_prob

        return opt_policy

    def evaluate_reward(self, policy):
        """
        apply MC method to approximate the reward
        """
        rew = 0
        state_mat = np.random.uniform(
            0, 1, size=(self.num_trials_for_eval, self.state_dim)
        )
        group_id_mat=np.floor(np.random.uniform(
            0, self.group_num, size=(self.num_trials_for_eval)
        ))#change this to get average vs worst case performance
        feature_tensor = []
        action_mat = []
        for index in range(self.num_trials_for_eval):
            state = state_mat[index, :]
            group_id=group_id_mat[index]
            action_prob = policy(state,group_id)
            assert np.all(action_prob >= 0.0) and np.allclose(
                np.sum(action_prob), 1.0
            ), "The policy is invalid."

            action_mat.append(action_prob)
            feature_mat = [
                self.feature_func(state, act_index, group_id)
                for act_index in range(self.action_num)
            ]
            feature_mat = np.stack(feature_mat, axis=0)
            feature_tensor.append(feature_mat)

        # feature_tensor has the shape of num * num_action * d
        feature_tensor = np.stack(feature_tensor, axis=0, dtype=np.float32)
        # reward_mat has the shape of num * num_action
        reward_mat = feature_tensor @ self.reward_param
        # action_mat has the shape of num * num_action
        action_mat = np.stack(action_mat, axis=0)

        rew = np.sum(np.multiply(reward_mat, action_mat)) / self.num_trials_for_eval

        return rew
    
    def evaluate_reward_group_wise(self, policy,states:Union[np.ndarray, list, None]):
        """
        apply MC method to approximate the reward
        """
        rewards = []
        state_mat = np.random.uniform(
            0, 1, size=(self.num_trials_for_eval, self.state_dim)
        )
        for group_id in range(self.group_num):
            feature_tensor = []
            action_mat = []
            for index in range(self.num_trials_for_eval):
                state = state_mat[index, :]
                action_prob = policy(state,group_id)
                assert np.all(action_prob >= 0.0) and np.allclose(
                    np.sum(action_prob), 1.0
                ), "The policy is invalid."

                action_mat.append(action_prob)
                feature_mat = [
                    self.feature_func(state, act_index, group_id)
                    for act_index in range(self.action_num)
                ]
                feature_mat = np.stack(feature_mat, axis=0)
                feature_tensor.append(feature_mat)

            # feature_tensor has the shape of num * num_action * d
            feature_tensor = np.stack(feature_tensor, axis=0, dtype=np.float32)
            # reward_mat has the shape of num * num_action
            reward_mat = feature_tensor @ self.reward_param
            # action_mat has the shape of num * num_action
            action_mat = np.stack(action_mat, axis=0)
            #print(reward_mat,action_mat)
            
            kendall_tau_distance = calculate_kendall_tau_distance(reward_mat, action_mat)

            print(f"Kendall Tau Distance: {kendall_tau_distance}")

            rew = np.sum(np.multiply(reward_mat, action_mat)) / self.num_trials_for_eval
            rewards.append(float(rew))

        return rewards
    

class GroupLinearBanditSep:
    def __init__(
        self,
        state_dim: int,
        action_num: int,
        group_num: int,
        reward_param: np.ndarray,
        feature_func,
        num_trials_for_eval: int = None,
        eval_metric: str='expectation',
        eval_metric_prob: str='KL'
    ) -> None:
        self.state_dim = state_dim
        self.action_num = action_num
        self.group_num = group_num
        self.action_space = [action_idx for action_idx in range(action_num)]
        self.group_ids=[group_id for group_id in range(group_num)]
        self.reward_param = reward_param
        self.feature_func = feature_func
        self.cur_state = np.random.uniform(0, 1, self.state_dim)
        self.eval_metric = eval_metric
        self.num_trials_for_eval = num_trials_for_eval
        self.eval_metric_prob=eval_metric_prob

    def reset(self) -> np.ndarray:
        self.cur_state = np.random.uniform(0, 1, self.state_dim)
        return self.cur_state

    def sample(self, action,group_id) -> float:
        assert action in self.action_space, "The input action is invalid."
        feature = self.feature_func(self.cur_state, action, group_id)
        reward_param=self.reward_param[group_id,:].ravel()
        #print(feature,reward_param)
        assert np.shape(feature) == np.shape(
            reward_param
        ), "The feature is invalid."
        rew = np.dot(feature, reward_param)
        return rew

    def get_opt_policy(self):
        def opt_policy(state: np.ndarray, group_id: int):
            # compute the optimal policy by enumerating the action space
            feature_mat = np.array(
                [
                    self.feature_func(state, action_idx, group_id)
                    for action_idx in range(self.action_num)
                ],
                dtype=np.float32,
            )
            reward_param=self.reward_param[group_id,:].ravel()
            assert np.shape(feature_mat) == (
                self.action_num,
                reward_param.size,
            ), "The feature matrix is invalid."
            rew_vec = np.matmul(feature_mat, reward_param)
            optimal_action = np.argmax(rew_vec)
            action_prob = np.zeros(self.action_num, np.float32)
            action_prob[optimal_action] = 1.0

            return action_prob

        return opt_policy
    
    def get_true_policy(self):
        def true_policy(state: np.ndarray, group_id: int):
            # compute the optimal policy by enumerating the action space
            feature_mat = np.array(
                [
                    self.feature_func(state, action_idx, group_id)
                    for action_idx in range(self.action_num)
                ],
                dtype=np.float32,
            )
            reward_param=self.reward_param[group_id,:].ravel()
            assert np.shape(feature_mat) == (
                self.action_num,
                reward_param.size,
            ), "The feature matrix is invalid."
            rew_vec = np.matmul(feature_mat, reward_param)
            
            action_prob = softmax(rew_vec)

            return action_prob

        return true_policy

    def evaluate_reward(self, policy):
        """
        apply MC method to approximate the reward
        """
        rew = 0
        state_mat = np.random.uniform(
            0, 1, size=(self.num_trials_for_eval, self.state_dim)
        )
        group_id_mat=np.floor(np.random.uniform(
            0, self.group_num, size=(self.num_trials_for_eval)
        ))#change this to get average vs worst case performance
        feature_tensor = []
        action_mat = []
        param_mat=[]
        for index in range(self.num_trials_for_eval):
            state = state_mat[index, :]
            group_id=group_id_mat[index]
            action_prob = policy(state,group_id)
            assert np.all(action_prob >= 0.0) and np.allclose(
                np.sum(action_prob), 1.0
            ), "The policy is invalid."
            reward_param=self.reward_param[group_id,:].ravel()
            param_mat.append(reward_param)
            action_mat.append(action_prob)
            feature_mat = [
                self.feature_func(state, act_index, group_id)
                for act_index in range(self.action_num)
            ]
            feature_mat = np.stack(feature_mat, axis=0)
            feature_tensor.append(feature_mat)


        # feature_tensor has the shape of num * num_action * d
        feature_tensor = np.stack(feature_tensor, axis=0, dtype=np.float32)
        # param_mat has the shape of num * d
        param_mat = np.stack(param_mat, axis=0)
        # reward_mat has the shape of num * num_action
        reward_mat = np.einsum('ijk,ik->ij', feature_tensor, param_mat)
        # action_mat has the shape of num * num_action
        action_mat = np.stack(action_mat, axis=0)

        rew = np.sum(np.multiply(reward_mat, action_mat)) / self.num_trials_for_eval

        return rew
    
    def evaluate_reward_group_wise(self, policy, states:Union[np.ndarray, list, None]):
        """
        apply MC method to approximate the reward
        """
        rewards = []

        if states is None:
            state_mat = np.random.uniform(
                0, 1, size=(self.num_trials_for_eval, self.state_dim)
            )
        elif isinstance(states, list):
            assert isinstance(states[0], GroupTransition),\
                f'expected list of GroupTransition not list of {type(states[0])}'
            state_mat, _, _, _ = process_pref_grp_data(states)    
        else:
            raise NotImplementedError()  


        for group_id in range(self.group_num):
            feature_tensor = []
            action_mat = []
            for index in range(self.num_trials_for_eval):
                state = state_mat[index, :]
                action_prob = policy(state,group_id)
                
                if self.eval_metric=='argmax':
                    # Modify action_prob to put 1 at argmax prob and 0 everywhere
                    max_prob_index = np.argmax(action_prob)
                    action_prob[:] = 0
                    action_prob[max_prob_index] = 1

                assert np.all(action_prob >= 0.0) and np.allclose(
                    np.sum(action_prob), 1.0
                ), "The policy is invalid."
                action_mat.append(action_prob)
                feature_mat = [
                    self.feature_func(state, act_index, group_id)
                    for act_index in range(self.action_num)
                ]
                feature_mat = np.stack(feature_mat, axis=0)
                feature_tensor.append(feature_mat)

            # feature_tensor has the shape of num * num_action * d
            feature_tensor = np.stack(feature_tensor, axis=0, dtype=np.float32)
            # reward_mat has the shape of num * num_action
            reward_mat = feature_tensor @ self.reward_param[group_id,:]
            # action_mat has the shape of num * num_action
            action_mat = np.stack(action_mat, axis=0)
            #print(reward_mat,action_mat)
            
            kendall_tau_distance = calculate_kendall_tau_distance(reward_mat, action_mat)

            #print(f"Kendall Tau Distance: {kendall_tau_distance}")

            rew = np.sum(np.multiply(reward_mat, action_mat)) / self.num_trials_for_eval
            rewards.append(float(rew))

        return rewards

    def evaluate_KL_group_wise(self, policy, states:Union[np.ndarray, list, None]):
        """
        apply MC method to approximate the reward
        """
        KLs = []

        if states is None:
            state_mat = np.random.uniform(
                0, 1, size=(self.num_trials_for_eval, self.state_dim)
            )
        elif isinstance(states, list):
            assert isinstance(states[0], GroupTransition),\
                f'expected list of GroupTransition not list of {type(states[0])}'
            state_mat, _, _, _ = process_pref_grp_data(states)    
        else:
            raise NotImplementedError()  

        true_policy=self.get_true_policy()
        for group_id in range(self.group_num):
            feature_tensor = []
            kl_mat = []
            for index in range(self.num_trials_for_eval):
                state = state_mat[index, :]
                action_prob = policy(state,group_id)
                action_prob_true = true_policy(state,group_id)
                if self.eval_metric_prob=='KL':
                    #print(action_prob_true,action_prob,'kl_before')
                    kl_distance=entropy(action_prob_true,action_prob)
                 
               
                kl_mat.append(kl_distance)
               

         
            kl_mat = np.stack(kl_mat, axis=0)
            #print(reward_mat,action_mat)
            
           
            #print(f"Kendall Tau Distance: {kendall_tau_distance}")

            rew = np.sum(kl_mat) / self.num_trials_for_eval
            KLs.append(float(rew))

        return KLs


def ret_feature_func(num_action: int, state_dim: int, group_num: int, feature_type: str):
    """
    return the feature function for an arbitrary number of actions and any state dimension.
    """

    def feature_func(state: np.ndarray, action: int, group_id: int) -> np.ndarray:
        assert action in range(num_action), "The input action is invalid."
        assert group_id in range(group_num), f'{group_id}'

        dim = 2 * state_dim
        feature = np.zeros(dim)
        if feature_type=='same':
            if group_id%2==0:
                for idx in range(state_dim):
                    feature[2 * idx] = (action/num_action + 1) * np.cos(state[idx] * np.pi)###maybe add group_id related qty
                    feature[2 * idx + 1] = (1.0 / (action/num_action + 1)) * np.sin(state[idx] * np.pi)
            else:
                for idx in range(state_dim):
                    feature[2 * idx] = (action/num_action + 1) * np.cos(state[idx] * np.pi)###maybe add group_id related qty
                    feature[2 * idx+1] = (1.0 / (action/num_action + 1)) * np.sin(state[idx] * np.pi)
                    #feature[2 * idx] = (action + 1) * np.sin(state[idx] * np.pi)
                    #feature[2 * idx + 1] = (1.0 / (action + 1)) * np.cos(state[idx] * np.pi)
        elif feature_type=='swapped':
            if group_id%2==0:
                for idx in range(state_dim):
                    feature[2 * idx] = (action/num_action + 1) * np.cos(state[idx] * np.pi)###maybe add group_id related qty
                    feature[2 * idx + 1] = (1.0 / (action/num_action + 1)) * np.sin(state[idx] * np.pi)
            else:
                for idx in range(state_dim):
                    feature[2 * idx+1] = (action/num_action + 1) * np.cos(state[idx] * np.pi)###maybe add group_id related qty
                    feature[2 * idx] = (1.0 / (action/num_action + 1)) * np.sin(state[idx] * np.pi)
                    #feature[2 * idx] = (action + 1) * np.sin(state[idx] * np.pi)
                    #feature[2 * idx + 1] = (1.0 / (action + 1)) * np.cos(state[idx] * np.pi)
        elif feature_type=='flipped':
            if group_id%2==0:
                for idx in range(state_dim):
                    feature[2 * idx] = (action/num_action + 1) * np.cos(state[idx] * np.pi)###maybe add group_id related qty
                    feature[2 * idx + 1] = (1.0 / (action/num_action + 1)) * np.sin(state[idx] * np.pi)
            else:
                for idx in range(state_dim):
                    #feature[2 * idx] = (action + 1) * np.cos(state[idx] * np.pi)###maybe add group_id related qty
                    #feature[2 * idx+1] = (1.0 / (action + 1)) * np.sin(state[idx] * np.pi)
                    feature[2 * idx] = (action/num_action + 1) * np.sin(state[idx] * np.pi)
                    feature[2 * idx + 1] = (1.0 / (action/num_action + 1)) * np.cos(state[idx] * np.pi)
        else:
            raise NotImplementedError

        return feature

    return feature_func


def ret_feature_func_debug(num_action: int, state_dim: int, group_num: int):
    """
    return the feature function for an arbitrary number of actions and any state dimension.
    """

    def feature_func(state: np.ndarray, action: int, group_id: int) -> np.ndarray:
        assert action in range(num_action), "The input action is invalid."
        assert group_id in range(group_num), f'{group_id}'

        dim = 2 * state_dim
        feature = np.zeros(dim)
        if group_id%2==0:
            for idx in range(state_dim):
                feature[2 * idx] = (action + 1) * np.cos(state[idx] * np.pi)###maybe add group_id related qty
                feature[2 * idx + 1] = (1.0 / (action + 1)) * np.sin(state[idx] * np.pi)
        else:
            for idx in range(state_dim):
                #feature[2 * idx+1] = (action/num_action + 1) * np.cos(state[idx] * np.pi)###maybe add group_id related qty
                #feature[2 * idx] = (1.0 / (action/num_action + 1)) * np.sin(state[idx] * np.pi)
                feature[2 * idx] = (action + 1) * np.sin(state[idx] * np.pi)
                feature[2 * idx + 1] = (1.0 / (action + 1)) * np.cos(state[idx] * np.pi)
                #feature[2 * idx+1] = (action + 1) * np.cos(state[idx] * np.pi)###maybe add group_id related qty
                #feature[2 * idx ] = (1.0 / (action + 1)) * np.sin(state[idx] * np.pi)
        

        return feature

    return feature_func

def calculate_kendall_tau_distance(reward_matrix, action_prob_matrix):
    # Get rankings based on rewards (higher reward is considered better)
    reward_rankings = [list(rank[::-1]) for rank in reward_matrix.argsort(axis=1)]

    # Get rankings based on action probabilities (higher probability is considered better)
    action_prob_rankings = [list(rank[::-1]) for rank in action_prob_matrix.argsort(axis=1)]

    # Calculate Kendall Tau distance
    distance, _ = kendalltau(reward_rankings, action_prob_rankings)

    return distance
