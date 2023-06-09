import numpy as np
class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        #returns the updated weights according to the basic gradient descent update scheme.
        return weight_tensor - self.learning_rate*gradient_tensor


