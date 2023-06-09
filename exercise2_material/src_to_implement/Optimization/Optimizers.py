import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        #returns the updated weights according to the basic gradient descent update scheme.
        return weight_tensor - self.learning_rate*gradient_tensor

class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        # self.momentum_gradient = np.zeros_like()
        self.in_init = True
    def calculate_update(self, weight_tensor, gradient_tensor):
        #returns the updated weights according to SgdWithMomentum update scheme
        if self.in_init == True:
            self.momentum_gradient = np.zeros_like(weight_tensor)
            self.in_init = False
        self.momentum_gradient = self.momentum_rate *self.momentum_gradient - self.learning_rate * gradient_tensor
        new_weight = weight_tensor + self.momentum_gradient
        return new_weight
class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.in_init = True
        self.iteration = 1
    def calculate_update(self, weight_tensor, gradient_tensor):
        #returns the updated weights according to Adam update scheme
        g = gradient_tensor
        if self.in_init == True:
            self.v_k = np.zeros_like(weight_tensor)
            self.r_k = np.zeros_like(gradient_tensor)
            self.in_init = False
        self.v_k = self.mu * self.v_k + (1-self.mu) * g
        self.r_k = self.rho * self.r_k + np.dot((1-self.rho)*g, g)
        self.v_hat = self.v_k/(1-self.mu**self.iteration)
        self.r_hat = self.r_k/(1-self.rho**self.iteration)
        new_weight = weight_tensor - self.learning_rate * (self.v_hat)/(np.sqrt(self.r_hat)+1e-8)
        self.iteration += 1
        return new_weight
    
if __name__=="__main__":
    optimizer = Adam(1., 0.01, 0.02)

    result = optimizer.calculate_update(1., 1.)
    np.testing.assert_almost_equal(result, np.array([0.]),
                                    err_msg="Possible reason: Formula for ADAM is not implemented correctly. Make"
                                            "sure that r and v are initialized with zeros.")

    result = optimizer.calculate_update(result, .5)
    np.testing.assert_almost_equal(result, np.array([-0.9814473195614205]),
                                    err_msg="Possible reason: Formula for ADAM is not implemented correctly. If you"
                                            "are sure that the implementation is correct, try to set eps=1e-8")
    
