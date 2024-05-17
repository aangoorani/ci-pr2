from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer

import numpy as np


class Linear(Layer):
    def __init__(self, inputs: int, outputs: int, need_bias: bool = False, mode="xavier") -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        if self.need_bias:
            res = x.data @ self.weight.data + self.bias.data
            return Tensor(res, requires_grad=x.requires_grad, depends_on=x.depends_on)
        else:
            return Tensor(x.data @ self.weight.data, requires_grad=x.requires_grad, depends_on=x.depends_on)

    def initialize(self):
        "TODO: initialize weight by initializer function (mode)"
        self.weight = Tensor(
            data=initializer((self.inputs, self.outputs), self.initialize_mode),
            requires_grad=True
        )

        "TODO: initialize bias by initializer function (zero mode)"
        if self.need_bias:
            self.bias = Tensor(
                data=initializer((1, self.outputs), mode="zero"),
                requires_grad=True
            )

    def zero_grad(self):
        "TODO: implement zero grad"
        if self.weight is not None:
            self.weight.zero_grad()
        if self.bias is not None:
            self.bias.zero_grad()

    def parameters(self):
        "TODO: return weights and bias"
        return [self.weight] + ([self.bias] if self.need_bias else [])

    def __str__(self) -> str:
        return "linear - total param: {} - in: {}, out: {}".format(self.inputs * self.outputs, self.inputs,
                                                                   self.outputs)
