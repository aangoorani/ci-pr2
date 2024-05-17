from mytorch import Tensor
from mytorch.layer import Layer

import numpy as np

class AvgPool2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
        super()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        batch_size, in_height, in_width, in_channels = x.shape
        kernel_height, kernel_width = self.kernel_size

        out_height = (in_height - kernel_height + 2 * self.padding[0]) // self.stride[0] + 1
        out_width = (in_width - kernel_width + 2 * self.padding[1]) // self.stride[1] + 1

        out_tensor = Tensor(np.zeros((batch_size, out_height, out_width, in_channels)), True)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + kernel_height
                w_start = j * self.stride[1]
                w_end = w_start + kernel_width

                x_slice = x[:, h_start:h_end, w_start:w_end, :]
                out_tensor[:, i, j, :] = np.mean(x_slice.data, axis=(1, 2))

        return out_tensor
    
    def __str__(self) -> str:
        return "avg pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)
