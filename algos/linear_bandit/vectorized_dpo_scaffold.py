import numpy as np

from .agent import Agent
from .action_selection import get_actions_features, calc_feat_diff
from utils.utils import sigmoid, split_state_actions

class DirectPreferenceOptimization(Agent):
    def __init__(
        self, 
        config,
        feature_dim,
        feature_func,
        ref_policy,
    ):
        super().__init__(config, feature_dim, feature_func, ref_policy)

    def calc_log_ratio_diff(
        self,
        dataset: np.ndarray
    ) -> np.ndarray:

        states, actions = split_state_actions(self.state_dim, dataset)

        pref_actions, npref_actions, pref_feat, npref_feat = get_actions_features(
            self.feature_func,
            states,
            actions
        )

        feat_diff = calc_feat_diff(
            self.feature_func,
            states,
            actions
        )

        log_ratio_diff = self.reg_coef * (feat_diff @ self.param)

        return feat_diff, log_ratio_diff

    def update_step(
        self,
        dataset: np.ndarray
    ) -> float:

        feat_diff, log_ratio_diff = self.calc_log_ratio_diff(dataset)

        coef = sigmoid(-log_ratio_diff)[:, None]
        neg_cur_data_grad = self.reg_coef * coef * feat_diff
        
        grad = -neg_cur_data_grad.mean(axis=0)

        self.hist_grad_squared_norm += np.sum(np.square(grad))
        if self.is_adaptive:
            step_size = self.ada_coef / np.sqrt(self.hist_grad_squared_norm)
        else:
            step_size = self.step_size
        self.param = self.param - step_size * grad

        return np.sqrt(np.sum(np.square(grad)))
    
    def evaluate_loss(self, dataset: np.ndarray) -> float:
        """
        Evaluate the loss on the dataset for any policy.
        """

        feat_diff, log_ratio_diff = self.calc_log_ratio_diff(dataset)
        loss = -np.log(sigmoid(log_ratio_diff)).mean()

        return loss