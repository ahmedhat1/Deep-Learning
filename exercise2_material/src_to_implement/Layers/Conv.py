from Layers.Base import BaseLayer
import numpy as np
from scipy import signal
from scipy.ndimage import convolve
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
        if (len(self.convolution_shape) == 3):
            self.img_2d = True
            self.n = self.convolution_shape[2]
            self.weights = np.random.uniform(0, 1, ((self.num_kernels, self.input_channel_num, self.m, self.n)))
            self.bias = np.random.uniform(0, 1, (self.num_kernels))
        else:
            self.weights = np.random.uniform(0, 1, ((self.num_kernels, self.input_channel_num, self.m)))
            self.bias = np.random.uniform(0, 1, (self.num_kernels))
    
    def forward(self, input_tensor):
        num_kernels = self.num_kernels
        if not self.img_2d:
            batch_size = input_tensor.shape[0]
            input_size = input_tensor.shape[2]
            filter_size = self.weights.shape[1]
            # num_kernels = self.weights.shape[0]
            output_size = (input_size - filter_size) // self.stride_shape[0] + 1

            self.output = np.zeros((batch_size, num_kernels, output_size ))

            for j in range(batch_size):
                for i in range(self.output.shape[2]):
                    for k in range(self.output.shape[1]):
                        input_slice = input_tensor[j][:,i:i+filter_size]
                        self.output[j][k][i] = np.sum(self.weights[k]* input_slice   )

            return self.output
        else:
            input_height, input_width = input_tensor.shape[:2]
            filter_height, filter_width = self.weights.shape[:2]
            num_filters = self.weights.shape[0]
            padding_height, padding_width = (0,0)
            stride_height, stride_width = self.stride_shape

            output_height = (input_height - filter_height + 2 * padding_height) // stride_height + 1
            output_width = (input_width - filter_width + 2 * padding_width) // stride_width + 1

            output = np.zeros((output_height, output_width, num_filters))

            padded_input = np.pad(input_tensor, ((padding_height, padding_height), (padding_width, padding_width), (0, 0)), mode='constant')

            for i in range(output_height):
                for j in range(output_width):
                    input_slice = padded_input[i*stride_height:i*stride_height+filter_height,
                                            j*stride_width:j*stride_width+filter_width]
                    output[i, j] = np.sum(input_slice * self.weights, axis=(0, 1))

            return output
    
    def backward(self, error_tensor):
        pass
    def initialize(self, weights_initializer, bias_initializer):

        if (self.img_2d == True):
            self.fan_in = self.input_channel_num * self.m * self.n
            self.fan_out = self.num_kernels * self.m * self.n
        else:
            self.fan_in = self.input_channel_num * self.m
            self.fan_out = self.num_kernels * self.m

        self.weights = weights_initializer.initialize(self.weights.shape, self.fan_in, self.fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, self.fan_in, self.fan_out)
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

