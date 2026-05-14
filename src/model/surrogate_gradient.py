"""
Surrogate gradient for backpropagation through non-differentiable spike functions.

Implements the rectangular surrogate: the backward pass substitutes a
unit-width rectangular window of height 1/a for the Heaviside step, allowing
gradients to flow through LIF neurons during training.
"""

import torch
import torch.nn as nn
import snntorch as snn

# =============================================================================
# Rectangular surrogate gradient
# =============================================================================

class RectangularSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, a):
        # In snntorch, input_ is already centered on the threshold: (v(t) - v_thr).
        ctx.save_for_backward(input_)
        ctx.a = a

        # Emit a spike (1.0) when the input crosses the threshold (input_ > 0).
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        a = ctx.a

        # Surrogate gradient: h(v) = 1/a if |v - v_thr| < a/2, else 0.
        mask = (torch.abs(input_) < (a / 2.0)).float()
        grad_input = grad_output * (1.0 / a) * mask

        return grad_input, None

def rectangular_sg(a=0.5):
    """Bind the surrogate width `a` and return the spike function."""
    def inner(x):
        return RectangularSurrogate.apply(x, a)
    return inner
