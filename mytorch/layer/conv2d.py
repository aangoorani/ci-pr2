from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer

import numpy as np

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), need_bias: bool = False, mode="xavier") -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        batch_size, in_height, in_width = x.shape
        kernel_height, kernel_width = self.kernel_size

        out_height = (in_height - kernel_height + 2 * self.padding[0]) // self.stride[0] + 1
        out_width = (in_width - kernel_width + 2 * self.padding[1]) // self.stride[1] + 1

        if self.padding != (0, 0):
            x = np.pad(x.data, ((0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1]), (0, 0)))

        out_tensor = Tensor(np.zeros((batch_size, out_height, out_width, self.out_channels)), True)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + kernel_height
                w_start = j * self.stride[1]
                w_end = w_start + kernel_width

                x_slice = x[:, h_start:h_end, w_start:w_end, :]
                out_tensor[:, i, j, :] = np.tensordot(x_slice.data, self.weight.data, axes=((1, 2, 3), (0, 1, 2))) + (self.bias.data if self.need_bias else 0)

        return out_tensor
    
    
    def initialize(self):
        print(f"kernel size: {self.kernel_size}")
        self.weight = initializer((self.kernel_size[0], self.kernel_size[1], self.in_channels, self.out_channels), self.initialize_mode)
        if self.need_bias:
            self.bias = Tensor(np.zeros((self.out_channels,)))

    def zero_grad(self):
        if self.weight is not None:
            self.weight.zero_grad()
        if self.bias is not None:
            self.bias.zero_grad()

    def parameters(self):
        "TODO: return weights and bias"
        return [self.weight] + ([self.bias] if self.need_bias else [])
    
    def __str__(self) -> str:
        return "conv 2d - total params: {} - kernel: {}, stride: {}, padding: {}".format(
                                                                                    self.kernel_size[0] * self.kernel_size[1],
                                                                                    self.kernel_size,
                                                                                    self.stride, self.padding)
