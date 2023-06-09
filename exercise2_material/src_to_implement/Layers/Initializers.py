import numpy as np
class Constant:
    def __init__(self, weight_constant=0.1):
        self.weight_constant = weight_constant
    def initialize(self,weights_shape,fan_in, fan_out):
        weights = np.full(weights_shape,self.weight_constant)
        return weights

class UniformRandom:
    def __init__(self):
        pass
    def initialize(self,weights_shape,fan_in, fan_out):
        weights=np.random.uniform(0,1,(weights_shape))
        return weights

class Xavier:
    def __init__(self):
        pass
    def initialize(self,weights_shape,fan_in, fan_out):
        sigma = np.sqrt(2/ (fan_in + fan_out))
        weights = np.random.normal(0, sigma, weights_shape)
        return weights

class He:
    def __init__(self):
        pass
    def initialize(self,weights_shape,fan_in, fan_out):
        sigma = np.sqrt(2/ fan_in )
        weights = np.random.normal(0, sigma, weights_shape)
        return weights
