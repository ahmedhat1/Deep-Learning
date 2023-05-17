import numpy as np
from Layers.Base import BaseLayer
class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
    def forward(self, input_tensor): 
        self.input_tensor = input_tensor
        self.output_tensor = np.maximum(0, input_tensor)
        return self.output_tensor
    def backward(self, error_tensor):
        x = self.input_tensor
        dx = error_tensor * (x>=0)
        return dx