import numpy as np
from typing import List
import ast

def softmax(arr: np.ndarray) -> np.ndarray:
    assert len(np.shape(arr)) == 1, "The input array is not 1-dim."
    softmax_arr = np.exp(arr - np.max(arr))
    softmax_arr = softmax_arr / np.sum(softmax_arr)
    return softmax_arr


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def return_apt_weights(weights: str, group_num: int) -> List:
    if weights=='equal':
        weights=np.ones(group_num)/group_num
        weights=weights.tolist()
    else:
        if group_num==1:
            return [1]
        weights=ast.literal_eval(weights)
    return weights