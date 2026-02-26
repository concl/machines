
import numpy as np
import machine_learning.numpy_autograd.numpy_autograd as autograd


class Module:
    """Base class for all modules. Defines the basic structure for forward and backward passes."""
    def __init__(self):
        self.parameters = []  # List of parameters (Tensors) in the module
    
    def forward(self, *args):
        raise NotImplementedError
    
    def __call__(self, *args):
        return self.forward(*args)
    
class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Initialize weights and bias
        self.W = autograd.Tensor(np.random.randn(in_features, out_features) * 0.01, requires_grad=True)
        self.b = autograd.Tensor(np.zeros(out_features), requires_grad=True)
        self.parameters.extend([self.W, self.b])
    
    def forward(self, x):
        return autograd.MatMul.apply(x, self.W) + self.b
    


class FFN(Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.linear1 = Linear(in_features, hidden_features)
        self.linear2 = Linear(hidden_features, out_features)
        self.parameters.extend(self.linear1.parameters)
        self.parameters.extend(self.linear2.parameters)
    
    def forward(self, x):
        x = self.linear1(x)
        x = autograd.ReLU.apply(x)
        x = self.linear2(x)
        return x
