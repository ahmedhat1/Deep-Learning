import numpy as np
from .Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_height, self.stride_width = stride_shape
        self.pool_height, self.pool_width = pooling_shape
        self.input_tensor = None
        self.output_height = None
        self.output_width = None
        self.batch_size = None
        self.channels = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.batch_size, self.channels, height, width = input_tensor.shape

        # Calculate output dimensions
        self.output_height = (height - self.pool_height) // self.stride_height + 1
        self.output_width = (width - self.pool_width) // self.stride_width + 1

        output_tensor = np.zeros((self.batch_size, self.channels, self.output_height, self.output_width))

        for b in range(self.batch_size):
            for ch in range(self.channels):
                for h in range(self.output_height):
                    for w in range(self.output_width):
                        h_start = h * self.stride_height
                        h_end = h_start + self.pool_height
                        w_start = w * self.stride_width
                        w_end = w_start + self.pool_width

                        pool_region = input_tensor[b, ch, h_start:h_end, w_start:w_end]
                        output_tensor[b, ch, h, w] = np.max(pool_region)

        return output_tensor

    def backward(self, error_tensor):
        grad_input = np.zeros_like(self.input_tensor)

        for b in range(self.batch_size):
            for ch in range(self.channels):
                for h in range(self.output_height):
                    for w in range(self.output_width):
                        h_start = h * self.stride_height
                        h_end = h_start + self.pool_height
                        w_start = w * self.stride_width
                        w_end = w_start + self.pool_width

                        pool_region = self.input_tensor[b, ch, h_start:h_end, w_start:w_end]
                        max_pos = np.unravel_index(np.argmax(pool_region), pool_region.shape)

                        grad_input[b, ch, h_start:h_end, w_start:w_end][max_pos] += error_tensor[b, ch, h, w]

        return grad_input
