import numpy as np
import autograd
import modules


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, parameters):
        self.parameters = parameters

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(param.data) for param in parameters]

    def step(self):
        for param, velocity in zip(self.parameters, self.velocities):
            if param.grad is not None:
                velocity *= self.momentum
                velocity += self.lr * param.grad
                param.data -= velocity

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None
