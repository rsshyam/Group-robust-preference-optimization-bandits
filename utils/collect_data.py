import collections
import numpy as np
import math
from typing import List
from envs.linear_bandit import LinearBandit
from envs.group_linear_bandit import GroupLinearBandit, GroupLinearBanditSep

Transition = collections.namedtuple(
    "Transition", ["state", "action_0", "action_1", "reward_0", "reward_1", "pref"]
)

GroupTransition = collections.namedtuple(
    "GroupTransition", ["state", "action_0", "action_1", "group_id", "reward_0", "reward_1", "pref"]
)

def sigmoid(x: float):
    return 1.0 / (1.0 + math.exp(-x))


def ret_uniform_policy(action_num: int = 0):
    assert action_num > 0, "The number of actions should be positive."

    def uniform_policy(state: np.ndarray = None):
        action_prob = np.full(shape=action_num, fill_value=1.0 / action_num)
        return action_prob

    return uniform_policy

def ret_uniform_policy_group(action_num: int = 0):
    assert action_num > 0, "The number of actions should be positive."

    def uniform_policy_group(state: np.ndarray = None, group_id: int = None):
        action_prob = np.full(shape=action_num, fill_value=1.0 / action_num)
        return action_prob

    return uniform_policy_group


def collect_preference_data(
    num: int, env: LinearBandit, policy_func
) -> List[Transition]:
    pref_dataset = []
    action_num = env.action_num
    for _ in range(num):
        state = env.reset()
        action_prob = policy_func(state)
        sampled_actions = np.random.choice(
            a=action_num, size=2, replace=False, p=action_prob  # replace=True
        )
        action_one, action_two = sampled_actions[0], sampled_actions[1]
        reward_one, reward_two = env.sample(action_one), env.sample(action_two)
        # print(state, reward_one, reward_two, reward_two - reward_one)

        bernoulli_param = sigmoid(reward_two - reward_one)
        # pref=1 means that the second action is preferred over the first one
        pref = np.random.binomial(1, bernoulli_param, 1)[0]
        transition = Transition(
            state, action_one, action_two, reward_one, reward_two, pref
        )
        pref_dataset.append(transition)
    return pref_dataset

def collect_group_preference_data(
    num: int, env: GroupLinearBandit, weights: List[float], policy_func
) -> List[GroupTransition]:
    pref_dataset = []
    action_num = env.action_num
    group_num = env.group_num
    print(weights)
    group_id_1=0
    group_id_2=0
    for _ in range(num):
        state = env.reset()
        group_id= np.random.choice(np.arange(group_num),1,p=np.array(weights))
        #print(group_id)
        if group_id==0:
            group_id_1+=1
        else:
            group_id_2+=1
        action_prob = policy_func(state,group_id)
        sampled_actions = np.random.choice(
            a=action_num, size=2, replace=False, p=action_prob  # replace=True
        )
        action_one, action_two = sampled_actions[0], sampled_actions[1]
        reward_one, reward_two = env.sample(action_one,group_id), env.sample(action_two,group_id)
        # print(state, reward_one, reward_two, reward_two - reward_one)

        bernoulli_param = sigmoid(reward_two - reward_one)
        # pref=1 means that the second action is preferred over the first one
        pref = np.random.binomial(1, bernoulli_param, 1)[0]
        group_transition = GroupTransition(
            state, action_one, action_two, group_id, reward_one, reward_two, pref
        )
        pref_dataset.append(group_transition)
    print(group_id_1,group_id_2)
    return pref_dataset


def collect_rl_data(num: int, env) -> List[float]:
    rl_dataset = []
    for _ in range(num):
        state = env.reset()
        rl_dataset.append(state)

    return rl_dataset


def merge_datasets(pref_dataset: List[Transition], rl_dataset: List[float]):
    merged_rl_dataset = rl_dataset
    for transition in pref_dataset:
        state = transition.state
        merged_rl_dataset.append(state)

    return merged_rl_dataset


def pref_to_rl(pref_dataset: List[Transition]):
    rl_dataset = []
    for transition in pref_dataset:
        state = transition.state
        rl_dataset.append(state)

    return rl_dataset
