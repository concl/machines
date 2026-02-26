
import numpy as np
import machine_learning.numpy_autograd.numpy_autograd as F

class Tensor:
    def __init__(self, data, requires_grad=False, backward_fn=None):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.backward_fn = backward_fn
        self.edges = []

    def __add__(self, other):
        assert isinstance(other, Tensor)
        result = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            backward_fn=self._backward_add
        )
        return result
    
    def _backward_add(self, grad):
        self.grad = grad
        self.edges[0].backward(grad)
        self.edges[1].backward(grad)

    def __mul__(self, other):
        assert isinstance(other, Tensor)
        result = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        return result
    
    def backward(self, grad=None):
        assert self.requires_grad, "Cannot call backward on a tensor that does not require gradients."
        assert self.data.size == 1, "Backward can only be called on scalar tensors."
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad


        for edge in self.edges:
            edge.backward(grad)


