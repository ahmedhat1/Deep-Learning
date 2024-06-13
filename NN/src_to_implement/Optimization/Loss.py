import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.input = None

    def forward(self, prediction_tensor, label_tensor):
        temp = np.log(prediction_tensor + np.finfo('float').eps)
        loss = np.sum(-temp * label_tensor)
        self.input = prediction_tensor
        return loss

    def backward(self, label_tensor):
        gradient = -(np.divide(label_tensor, self.input))
        return gradient
