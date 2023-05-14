import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size,optimizer=None,gradient_weights=None):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, size=(input_size+1, output_size))
        # self.bias = np.random.uniform(0, 1, size=output_size)
        self.__optimizer = optimizer
        self.__gradient_weights = gradient_weights

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # print("error_tensor shape", input_tensor.shape)
        # print("output_tensor shape", self.weights.shape)
        self.output_tensor = np.dot( input_tensor, self.weights[:-1]) +      self.weights[-1]
        return self.output_tensor

    def backward(self, error_tensor):

                                           
        error_tesnor_prev = np.dot(error_tensor, self.weights[:-1].T)
        # print("weights", self.weights[:-1].shape)
        # print("self.__gradient_weights", self.__gradient_weights.shape)
        self.__gradient_weights = np.concatenate((np.dot( error_tensor.T, self.input_tensor) , error_tensor),axis = 0)
        if self.optimizer:
            # bias = self.weights[-1]
            # print("weights", self.weights[:-1].shape)
            # print("self.__gradient_weights", self.__gradient_weights.shape)
            self.weights = self.__optimizer.calculate_update(self.weights, self.__gradient_weights.T)
            # new_bias = self.__optimizer.calculate_update(self.weights[-1], error_tensor)
            # self.weights = np.concatenate((self.weights, np.expand_dims(bias,axis=0)))
            # self.bias = self.__optimizer.calculate_update(self.bias, self.__gradient_weights)
        return error_tesnor_prev

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