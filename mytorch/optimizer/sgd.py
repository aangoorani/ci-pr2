from typing import List
from mytorch.layer import Layer
from mytorch.layer import Linear, Conv2d
from mytorch.optimizer import Optimizer
from mytorch.tensor import Tensor


class SGD(Optimizer):
    def __init__(self, layers:List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        "TODO: implement SGD algorithm"
        for l in self.layers:
            l.weight = l.weight - l.weight.grad * Tensor([self.learning_rate])
            
