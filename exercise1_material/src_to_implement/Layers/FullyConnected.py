# from Layers.Base import BaseLayer
# from numpy import random
# import numpy as np
# from Optimization.Optimizers import Sgd

# class FullyConnected(BaseLayer):
#     def __init__(self,input_size, output_size):
#         BaseLayer.__init__(self)
#         self.trainable = True
#         # size of self.weights = (input_size) (e.g. 128*128)
#         self.weights = random.uniform(low=0.0, high = 1.0, size =(output_size,input_size))
#         self.__optimizer = None
#         self.__gradient_weights = None
#         pass
    
#     #input tensor is a matrix with input size column and batch size rows. The batch size represents the 
#     # number of inputs processed simultaneously. The output size is a parameter of the layer specifying 
#     # the number of columns of the output.
#     def forward(self, input_tensor):
#         self.output = np.dot(input_tensor, np.transpose(self.weights))
#         return self.output

#     def backward(self,error_tensor):
#         gradient_tensor = np.dot(np.transpose(self.output),error_tensor)
#         # self.weights = Sgd(learning_rate)
#         return gradient_tensor


#     # @property
#     # def optimizer(self):
#     #     return self._optimizer

#     # @optimizer.setter
#     # def optimizer(self, value):
#     #     self.__optimizer = value

    
#     def __get_optimizer(self):
#         return self.__optimizer
#     def __set_optimizer(self, optim):
#         self.__optimizer = optim
#     optimizer = property(
#         fget = __get_optimizer,
#         fset = __set_optimizer,
#         doc = "optimizer property"
#     )
#     # @property
#     # def gradient_weights(self):
#     #     return self.optimizer.gradient_weights if self.optimizer else None
    
#     def __get_gradient_weights(self):
#         return self.__get_gradient_weights()

#     gradient_weights = property(
#         fget = __get_gradient_weights,
#         doc = "gradient weights"
#     )

import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size,optimizer=None,gradient_weights=None):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, size=(input_size+1,output_size))
        self.bias = np.random.uniform(0, 1, size=(output_size, 1))
        self.__optimizer = optimizer
        self.__gradient_weights = gradient_weights

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.dot(self.weights, input_tensor.T)

    def backward(self, error_tensor):
        if self.optimizer:
            gradient_weights = np.dot(error_tensor, self.input_tensor.T)
            self.weights = self.__optimizer.calculate_update(self.weights, gradient_weights)
        return np.dot(self.weights.T, error_tensor)

    def __get_optimizer(self):
        return self.__optimizer
    def __set_optimizer(self, optim):
        self.__optimizer = optim

    optimizer = property(
        fget = __get_optimizer,
        fset = __set_optimizer,
        doc = "optimizer property"
    )

    def __get_gradient_weights(self):
        return self.__gradient_weights
    gradient_weights = property(
        fget = __get_gradient_weights,
        doc = "gradient weights"
    )