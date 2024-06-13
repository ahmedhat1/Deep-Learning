import numpy as np
from scipy import signal
from .Base import BaseLayer

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        if len(convolution_shape) == 2:
            self.weights = np.random.uniform(0, 1, (num_kernels, convolution_shape[0], convolution_shape[1]))
        elif len(convolution_shape) == 3:
            self.weights = np.random.uniform(0, 1, (num_kernels, convolution_shape[0], convolution_shape[1], convolution_shape[2]))
        self.bias = np.random.uniform(0, 1, num_kernels)
        self._optimizer_w = None
        self._optimizer_b = None
        self.input_tensor = None
        self.output_tensor = None
        self.gradient_w = None
        self.gradient_b = None

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        self.input_tensor = input_tensor
        num_channels = input_tensor.shape[1]
        is_1d_conv = len(input_tensor.shape) == 3
        output_tensor = np.zeros([batch_size, self.num_kernels, *input_tensor.shape[2:]])

        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(num_channels):
                    output_tensor[b, k] += signal.correlate(input_tensor[b, c], self.weights[k, c], mode='same')
                output_tensor[b, k] += self.bias[k]

        if is_1d_conv:
            output_tensor = output_tensor[:, :, ::self.stride_shape[0]]
        else:
            output_tensor = output_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]

        self.output_tensor = output_tensor
        return output_tensor

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]
        num_channels = self.input_tensor.shape[1]
        propagated_error = np.zeros(self.input_tensor.shape)
        grad_weights = np.zeros(self.weights.shape)
        self.gradient_w = np.zeros_like(self.weights)
        self.gradient_b = np.zeros_like(self.bias)

        for b in range(batch_size):
            strided_error = np.zeros((self.num_kernels, *self.input_tensor.shape[2:]))
            is_2d_conv = len(error_tensor.shape) == 4

            for k in range(error_tensor.shape[1]):
                error_slice = error_tensor[b, k, :]
                if is_2d_conv:
                    strided_error[k, ::self.stride_shape[0], ::self.stride_shape[1]] = error_slice
                else:
                    strided_error[k, ::self.stride_shape[0]] = error_slice

            for c in range(num_channels):
                flipped_weights = np.flip(self.weights, 0)[:, c, :]
                error_convolution = signal.convolve(strided_error, flipped_weights, mode='same')
                center_channel = int(error_convolution.shape[0] / 2)
                propagated_error[b, c, :] = error_convolution[center_channel, :]

            for k in range(self.num_kernels):
                self.gradient_b[k] += np.sum(error_tensor[b, k, :])
                for c in range(num_channels):
                    input_channel_data = self.input_tensor[b, c, :]
                    if is_2d_conv:
                        pad_width = self.convolution_shape[1] / 2
                        pad_height = self.convolution_shape[2] / 2
                        pad_w = (int(np.floor(pad_width)), int(np.floor(pad_width - 0.5)))
                        pad_h = (int(np.floor(pad_height)), int(np.floor(pad_height - 0.5)))
                        padded_data = np.pad(input_channel_data, (pad_w, pad_h))
                    else:
                        pad_width = self.convolution_shape[1] / 2
                        pad_w = (int(np.floor(pad_width)), int(np.floor(pad_width - 0.5)))
                        padded_data = np.pad(input_channel_data, pad_w)

                    grad_weights[k, c, :] = signal.correlate(padded_data, strided_error[k, :], mode="valid")

            self.gradient_w += grad_weights

        if self._optimizer_w is not None:
            self.weights = self._optimizer_w.calculate_update(self.weights, self.gradient_w)
        if self._optimizer_b is not None:
            self.bias = self._optimizer_b.calculate_update(self.bias, self.gradient_b)

        return propagated_error

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    @property
    def optimizer(self):
        return self._optimizer_w

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer_w = optimizer

    @property
    def optimizer_b(self):
        return self._optimizer_b

    @optimizer_b.setter
    def optimizer_b(self, optimizer_b):
        self._optimizer_b = optimizer_b

    @property
    def gradient_weights(self):
        return self.gradient_w

    @property
    def gradient_bias(self):
        return self.gradient_b
