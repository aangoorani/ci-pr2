from mytorch import Tensor
from ..activation import softmax
import numpy as np


def  CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    _preds = softmax(preds)
    _sum = (label * _preds).sum()
    result_array = np.ndarray(preds.shape)
    result_array.fill(label.shape[0])
    size = Tensor(result_array)
    size = size**-1
    return _sum * size