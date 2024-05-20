from Layers.Base import BaseLayer
import numpy as np
from scipy import signal

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.weights = None
        self.bias = None
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.img_2d = False

        # self._optimizer_b = None
        # self._optimizer_w=None

        self.input_channel_num = self.convolution_shape[0]

        self.m = self.convolution_shape[1]

        # in case of 2D input
        if (len(convolution_shape) == 3):
            self.img_2d = True
            self.n = self.convolution_shape[2]
            self.weights = np.random.uniform(0, 1, ((self.num_kernels, self.input_channel_num, self.m, self.n)))
        else:
            self.weights = np.random.uniform(0, 1, ((self.num_kernels, self.input_channel_num, self.m)))
        self.bias = np.random.uniform(0, 1, (self.num_kernels))
    
    def initialize(self, weights_initializer, bias_initializer):

        if (self.img_2d == True):
            self.fan_in = self.input_channel_num * self.m * self.n
            self.fan_out = self.num_kernels * self.m * self.n
        else:
            self.fan_in = self.input_channel_num * self.m
            self.fan_out = self.num_kernels * self.m

        self.weights = weights_initializer.initialize(self.weights.shape, self.fan_in, self.fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, self.fan_in, self.fan_out)

    
    def forward(self, input_tensor):
        #– The input layout for 1D is defined in b, c, y order, for 2D in b, c, y, x order. Here,
        #b stands for the batch, c represents the channels and x, y represent the spatial dimensions
        self.input_tensor = input_tensor
        self.b = input_tensor[0]
        self.c = input_tensor[1]
        self.y = input_tensor[2]
        if self.img_2d:
            self.x = input_tensor[3]
        #self.output =

    def backward(self, error_tensor):
        pass

    # @property
    # def gradient_weights(self):
    #     return self.gradient_w

    # @property
    # def gradient_bias(self):
    #     return self.gradient_b

    # @property
    # def optimizer(self):
    #     return self._optimizer_w

    # @optimizer.setter
    # def optimizer(self, opt):
    #     self._optimizer_b = copy.deepcopy(opt)
    #     self._optimizer_w=copy.deepcopy(opt)

