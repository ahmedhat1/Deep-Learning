import numpy as np
from Layers.Base import BaseLayer
class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
    def forward(self, input_tensor):
        zeros_tensor = np.zeros_like(input_tensor)
        # if np.greater(input_tensor, zeros_tensor).any():
        if np.any(input_tensor > 0 ):
            input_tensor = np.subtract(input_tensor, np.max(input_tensor))
        exp_tensor = np.exp(input_tensor)
        try:
            exp_tensor.shape[1]
            den = np.sum(exp_tensor, axis = 0)
            den = np.repeat(np.expand_dims(den,axis=1),exp_tensor.shape[0],axis=1)
            self.output_tensor = exp_tensor/den.T
            print("outtesnro", np.sum(self.output_tensor))
            return self.output_tensor
        except:
            print("outtesnro", np.sum(self.output_tensor))

            return exp_tensor/np.sum(exp_tensor)
        # output = exp_tesnor[:,]
    def backward(self, error_tensor):
        pass