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
            self.bias = np.random.uniform(0, 1, (self.num_kernels))
        else:
            self.weights = np.random.uniform(0, 1, ((self.num_kernels, self.input_channel_num, self.m)))
            self.bias = np.random.uniform(0, 1, (self.num_kernels))
    
    def forward(self, input_tensor):
        pass
    
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

