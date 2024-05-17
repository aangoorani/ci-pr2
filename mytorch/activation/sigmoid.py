import numpy as np
from mytorch import Tensor, Dependency

def sigmoid(x: Tensor) -> Tensor:
    """
    TODO: implement sigmoid function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    shape = x.shape
    numerator = Tensor(np.ones(shape))
    denominator = Tensor(np.ones(shape)) + ((-x).exp())
    denominator = denominator**-1
    return numerator * denominator