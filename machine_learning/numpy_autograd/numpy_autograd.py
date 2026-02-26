
import numpy as np

class AutoGradFunction:
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class AddFunction(AutoGradFunction):
    @staticmethod
    def forward(ctx, a, b):
        return a + b
    
    @staticmethod
    def backward(ctx, grad_output):

        # The derivative of a + b with respect to a is 1,
        # and with respect to b is also 1
        return grad_output, grad_output

class MulFunction(AutoGradFunction):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b
    
    @staticmethod
    def backward(ctx, grad_output):

        # The derivative of a * b with respect to a is b, 
        # and with respect to b is a
        a, b = ctx.saved_tensors
        return grad_output * b, grad_output * a
    
class MatMul(AutoGradFunction):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b
    
    @staticmethod
    def backward(ctx, grad_output):

        # The derivative of a_ij with respect to the output is b_jk,
        # and the derivative of b_jk with respect to the output is a_ij
        # Using the chain rule, we get the partial for a_ij as:
        # grad_output_ik * b_jk and for b_jk as a_ij * grad_output_ik
        a, b = ctx.saved_tensors
        grad_a = grad_output @ b.T
        grad_b = a.T @ grad_output
        return grad_a, grad_b
    
class ReLU(AutoGradFunction):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.maximum(0, x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output * (x > 0).astype(x.dtype)
        return grad_input

class CrossEntropyLoss(AutoGradFunction):
    @staticmethod
    def forward(ctx, logits, labels):
        # Compute softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Compute cross-entropy loss
        batch_size = logits.shape[0]
        loss = -np.sum(labels * np.log(softmax_probs + 1e-15)) / batch_size
        
        ctx.save_for_backward(softmax_probs, labels)
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        softmax_probs, labels = ctx.saved_tensors
        batch_size = softmax_probs.shape[0]
        
        # Gradient of loss with respect to logits
        grad_logits = (softmax_probs - labels) / batch_size
        
        return grad_logits, None  # No gradient for labels

class AutoGradContext:
    def __init__(self):
        self.saved_tensors = None
    
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class Node:
    """A node in the autograd computation graph. 
    Each node wraps an AutoGradFunction and holds the context 
    and references to input tensors for backward traversal."""

    def __init__(self, function: AutoGradFunction, ctx: AutoGradContext, inputs: list):
        self.function = function    # The AutoGradFunction class (e.g. AddFunction)
        self.ctx = ctx              # Context holding saved tensors
        self.inputs = inputs        # List of input Tensors (for graph traversal)

    def backward(self, grad_output):
        return self.function.backward(self.ctx, grad_output)


class Tensor:
    """A wrapper around numpy arrays that tracks the computation graph
    for automatic differentiation."""

    def __init__(self, data, requires_grad=False, grad_fn: Node = None):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None            # Accumulated gradient
        self.grad_fn = grad_fn      # Node that produced this tensor (None for leaves)

    