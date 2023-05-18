import numpy as np
from Layers.Base import BaseLayer
class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
    def forward(self, input_tensor):
        maximum = np.expand_dims(np.max(input_tensor, 1), 1) 
        x_tilda = input_tensor - maximum # to increase stability
        numerator = np.exp(x_tilda) 
        denominator = np.expand_dims(np.sum(numerator, axis=1,), 1)
        self.output_tensor = numerator / denominator
        return self.output_tensor
    def backward(self, label_tensor):
        scalar_rows = np.sum(np.multiply(label_tensor,self.output_tensor), axis=1)
        error = np.multiply(self.output_tensor, np.subtract(label_tensor[0:label_tensor.shape[0], :].T, scalar_rows).T)
        return error

