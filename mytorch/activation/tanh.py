import numpy as np
from mytorch import Tensor, Dependency

def tanh(x: Tensor) -> Tensor:
    """
    TODO: (optional) implement tanh function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    numerator = x.exp() - (-x).exp()
    denominator = x.exp() + (-x).exp()
    denominator = denominator**-1
    return numerator * denominator