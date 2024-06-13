import numpy as np
from .Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()


    def forward(self, input_tensor):
        # Getting the shape of the input tensor
        self.shape = input_tensor.shape
        # When reshaping, we must take into consideration the batch size (self.shape[0])
        return np.array(input_tensor).reshape(self.shape[0], np.prod(self.shape[1:]))

    def backward(self, error_tensor):
        # reshaping the error tesnor again to the original shape
        return np.array(error_tensor).reshape(self.shape)
