
import numpy as np

def unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Sum grad over axes that were broadcasted to match target_shape."""
    while len(grad.shape) > len(target_shape):
        grad = grad.sum(axis=0)
    
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class AutogradFunction:
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class AddFunction(AutogradFunction):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a.shape, b.shape)
        return a + b
    
    @staticmethod
    def backward(ctx, grad_output):
        a_shape, b_shape = ctx.saved_tensors
        grad_output_a = unbroadcast(grad_output, a_shape)
        grad_output_b = unbroadcast(grad_output, b_shape)
        return grad_output_a, grad_output_b

class MulFunction(AutogradFunction):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b
    
    @staticmethod
    def backward(ctx, grad_output):

        # The derivative of a * b with respect to a is b, 
        # and with respect to b is a
        a, b = ctx.saved_tensors
        grad_output_a = unbroadcast(grad_output * b, a.shape)
        grad_output_b = unbroadcast(grad_output * a, b.shape)

        return grad_output_a, grad_output_b
    
class MatMul(AutogradFunction):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b
    
    @staticmethod
    def backward(ctx, grad_output):

        # The derivative of a_ij with respect to an element c_ik in the output is is b_jk,
        # and the derivative of b_jk with respect to an element c_ik in the output  is a_ij
        # Using the chain rule, we get the partial for a_ij as:
        # grad_output_ik * b_jk (summed over k) and for b_jk as a_ij * grad_output_ik (summed over i)
        a, b = ctx.saved_tensors

        a_T = np.swapaxes(a, -1, -2)
        b_T = np.swapaxes(b, -1, -2)

        grad_a = np.matmul(grad_output, b_T)
        grad_b = np.matmul(a_T, grad_output)
        return grad_a, grad_b
    
class ReLUFunction(AutogradFunction):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.maximum(0, x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output * (x > 0).astype(x.dtype)
        return grad_input

class CrossEntropyLossFunction(AutogradFunction):
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

class AutogradContext:
    def __init__(self):
        self.saved_tensors = None
    
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class AutogradNode:
    """A node in the autograd computation graph. 
    Each node wraps an AutoGradFunction and holds the context 
    and references to input tensors for backward traversal."""

    def __init__(self, function: AutogradFunction, ctx: AutogradContext, inputs: list):
        self.function = function    # The AutogradFunction class (e.g. AddFunction)
        self.ctx = ctx              # Context holding saved tensors
        self.inputs = inputs        # List of input Tensors (for graph traversal)

    def backward(self, grad_output):
        grad_output = self.function.backward(self.ctx, grad_output)
        for i, input_tensor in enumerate(self.inputs):
            if input_tensor.requires_grad:
                input_tensor.backward(grad_output[i] if isinstance(grad_output, tuple) else grad_output)
        
        return grad_output



class Tensor:
    """A wrapper around numpy arrays that tracks the computation graph
    for automatic differentiation."""

    def __init__(self, data, requires_grad=False, grad_fn: AutogradNode = None):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None          # Accumulated gradient
        self.grad_fn = grad_fn    # Node that produced this tensor (None for leaves)
    
    def backward(self, grad_output=None):
        if self.grad_fn is not None:
            if grad_output is None:
                grad_output = np.ones_like(self.data)
            grad_input = self.grad_fn.backward(grad_output)
            if self.requires_grad:
                self.grad = grad_input
            # Propagate gradients backward through the graph
            for input_tensor in self.grad_fn.inputs:
                input_tensor.backward(grad_input)
        else:
            # If no grad_fn, this is a leaf tensor and no backward pass is needed
            if self.requires_grad and grad_output is not None:
                self.grad = grad_output
        
    def __add__(self, other):
        if isinstance(other, Tensor):
            ctx = AutogradContext()
            result_data = AddFunction.forward(ctx, self.data, other.data)
            grad_fn = AutogradNode(AddFunction, ctx, [self, other])
            return Tensor(result_data, requires_grad=self.requires_grad or other.requires_grad, grad_fn=grad_fn)
        else:
            return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            ctx = AutogradContext()
            result_data = MulFunction.forward(ctx, self.data, other.data)
            grad_fn = AutogradNode(MulFunction, ctx, [self, other])
            return Tensor(result_data, requires_grad=self.requires_grad or other.requires_grad, grad_fn=grad_fn)
        else:
            return NotImplemented
    
    def __matmul__(self, other):
        if isinstance(other, Tensor):
            return matmul(self, other)
        else:
            return NotImplemented
    
    def __getitem__(self, key):
        return Tensor(self.data[key], requires_grad=self.requires_grad)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    ctx = AutogradContext()
    result_data = MatMul.forward(ctx, a.data, b.data)
    grad_fn = AutogradNode(MatMul, ctx, [a, b])
    return Tensor(result_data, requires_grad=a.requires_grad or b.requires_grad, grad_fn=grad_fn)