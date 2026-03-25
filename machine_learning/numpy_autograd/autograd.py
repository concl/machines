import numpy as np

# ----- Helper functions -----


def unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Sum grad over axes that were broadcasted to match target_shape."""
    while len(grad.shape) > len(target_shape):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


# ----- Autograd functions for basic operations -----


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


class MatMulFunction(AutogradFunction):
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
    """
    ReLU activation function with autograd support.
    Should be used as a module for abstraction.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.maximum(0, x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = grad_output * (x > 0).astype(x.dtype)
        return grad


class SoftmaxFunction(AutogradFunction):
    @staticmethod
    def forward(ctx, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        softmax = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        ctx.save_for_backward(softmax)
        return softmax

    @staticmethod
    def backward(ctx, grad_output):
        (softmax,) = ctx.saved_tensors
        grad_input = softmax * (
            grad_output - np.sum(grad_output * softmax, axis=-1, keepdims=True)
        )
        return grad_input


class CrossEntropyLossFunction(AutogradFunction):
    """
    Cross-entropy loss function with autograd support.
    Should be used as a module for abstraction.
    """

    @staticmethod
    def forward(ctx, logits, labels, eps=1e-9):
        # Compute softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Compute cross-entropy loss for integer labels
        batch_size = logits.shape[0]
        loss = -np.sum(np.log(softmax_probs + eps)[np.arange(batch_size), labels]) / batch_size

        ctx.save_for_backward(softmax_probs, labels)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        softmax_probs, labels = ctx.saved_tensors
        batch_size = softmax_probs.shape[0]

        # Gradient: softmax_probs - one_hot(labels)
        grad_logits = softmax_probs.copy()
        grad_logits[np.arange(batch_size), labels] -= 1
        grad_logits /= batch_size

        return grad_logits * grad_output, None


# ----- Autograd graph classes -----


class AutogradContext:
    def __init__(self):
        self.saved_tensors = None

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class AutogradNode:
    """
    A node in the autograd computation graph.
    Each node wraps an AutoGradFunction and holds the context
    and references to input tensors for backward traversal.
    """

    def __init__(self, function: AutogradFunction, ctx: AutogradContext, inputs: list):
        self.function = function  # The AutogradFunction class (e.g. AddFunction)
        self.ctx = ctx  # Context holding saved tensors
        self.inputs = inputs  # List of input Tensors (for graph traversal)

    def backward(self, grad_output):
        grads = self.function.backward(self.ctx, grad_output)
        for i, input_tensor in enumerate(self.inputs):
            grad = grads[i] if isinstance(grads, tuple) else grads
            if grad is None:
                continue
            if input_tensor.requires_grad:
                input_tensor.grad = grad if input_tensor.grad is None else input_tensor.grad + grad
            if input_tensor.grad_fn is not None:
                input_tensor.grad_fn.backward(grad)


# ----- Tensor class -----


class Tensor:
    """
    A wrapper around numpy arrays that tracks the computation graph
    for automatic differentiation.
    """

    def __init__(
        self, data, requires_grad=False, grad_fn: AutogradNode = None, dtype=np.float32
    ):
        self.data = np.array(data, dtype=data.dtype if isinstance(data, np.ndarray) else dtype)
        self.requires_grad = requires_grad
        self.grad = None  # Accumulated gradient
        self.grad_fn = grad_fn  # Node that produced this tensor (None for leaves)

    def backward(self, grad_output=None):
        if self.grad_fn is not None:
            if grad_output is None:
                grad_output = np.ones_like(self.data)
            self.grad_fn.backward(grad_output)
        else:
            if self.requires_grad and grad_output is not None:
                self.grad = grad_output

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        if isinstance(other, Tensor):
            ctx = AutogradContext()
            result_data = AddFunction.forward(ctx, self.data, other.data)
            grad_fn = AutogradNode(AddFunction, ctx, [self, other])
            return Tensor(
                result_data,
                requires_grad=self.requires_grad or other.requires_grad,
                grad_fn=grad_fn,
            )
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Tensor):
            ctx = AutogradContext()
            result_data = MulFunction.forward(ctx, self.data, other.data)
            grad_fn = AutogradNode(MulFunction, ctx, [self, other])
            return Tensor(
                result_data,
                requires_grad=self.requires_grad or other.requires_grad,
                grad_fn=grad_fn,
            )
        else:
            return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            return matmul(self, other)
        else:
            return NotImplemented

    def __getitem__(self, key):
        return Tensor(self.data[key], requires_grad=self.requires_grad)
    
    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"


# ----- Autograd function wrappers for user-friendly API -----


def matmul(a: Tensor, b: Tensor) -> Tensor:
    ctx = AutogradContext()
    result_data = MatMulFunction.forward(ctx, a.data, b.data)
    grad_fn = AutogradNode(MatMulFunction, ctx, [a, b])
    return Tensor(
        result_data, requires_grad=a.requires_grad or b.requires_grad, grad_fn=grad_fn
    )


def relu(x: Tensor) -> Tensor:
    ctx = AutogradContext()
    result_data = ReLUFunction.forward(ctx, x.data)
    grad_fn = AutogradNode(ReLUFunction, ctx, [x])
    return Tensor(result_data, requires_grad=x.requires_grad, grad_fn=grad_fn)


def softmax(x: Tensor) -> Tensor:
    ctx = AutogradContext()
    result_data = SoftmaxFunction.forward(ctx, x.data)
    grad_fn = AutogradNode(SoftmaxFunction, ctx, [x])
    return Tensor(result_data, requires_grad=x.requires_grad, grad_fn=grad_fn)


def cross_entropy_loss(logits: Tensor, labels: Tensor) -> Tensor:
    ctx = AutogradContext()
    loss_data = CrossEntropyLossFunction.forward(ctx, logits.data, labels.data.flatten().astype(np.intp))
    grad_fn = AutogradNode(CrossEntropyLossFunction, ctx, [logits, labels])
    return Tensor(loss_data, requires_grad=logits.requires_grad, grad_fn=grad_fn)
