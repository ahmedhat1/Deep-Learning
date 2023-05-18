from Optimization import Optimizers
import copy
class NeuralNetwork():
    def __init__(self,optimizer):
        self.optimizer=optimizer
        self.loss=[]#contains loss values after calling train()
        self.layers=[] #contains all layers
        self.data_layer=None #contains input data & label
        self.loss_layer=None # contain cross entropy layer

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        self.output_tensor=self.input_tensor
        for layer in self.layers:
            self.output_tensor = layer.forward(self.output_tensor)

        self.output_tensor=self.loss_layer.forward(self.output_tensor,self.label_tensor)

        return self.output_tensor
    def backward(self):
        self.error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            self.error_tensor = layer.backward(self.error_tensor)

    def append_layer(self,layer):
        if layer.trainable:
            layer.__set_optimizer =copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self,iterations):
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self,input_tensor):
        output = input_tensor
        for lay in self.layers:
            output = lay.forward(output)
        return output