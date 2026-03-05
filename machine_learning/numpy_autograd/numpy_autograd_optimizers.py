
from numpy_autograd import *

class Optimizer:
    """Base class for all optimizers. Defines the basic structure for parameter updates."""
    def __init__(self, parameters):
        self.parameters = parameters  # List of parameters (Tensors) to optimize
    
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters)
        self.lr = lr
    
    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad


