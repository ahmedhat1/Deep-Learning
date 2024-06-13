import numpy as np
from .Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        exps = np.exp(input_tensor - input_tensor.max())
        self.output_tensor = exps / np.expand_dims(np.sum(exps, axis=1), 1)
        return self.output_tensor

    def backward(self, error_tensor):
        EnYhat = np.sum(np.multiply(error_tensor, self.output_tensor), axis=1)
        error = np.multiply(self.output_tensor, np.subtract(error_tensor.T, EnYhat).T)
        return error
