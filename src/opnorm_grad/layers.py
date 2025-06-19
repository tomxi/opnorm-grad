import torch
import torch.nn as nn

class OpNormConv1d(nn.Conv1d):
    """
    A 1D convolutional layer that normalizes gradients by sequence length.

    This layer extends `torch.nn.Conv1d` and scales the gradients of its weights
    and biases during the backward pass. The scaling factor is the length of
    the output sequence, which corresponds to the number of times the
    convolutional filter is applied. This technique helps stabilize training for
    models that process variable-length sequences, where gradient magnitudes
    could otherwise depend on input length.

    Since this class inherits from `nn.Conv1d`, it can be used as a drop-in
    replacement.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This attribute is for caching the output length for the backward hook.
        self.L_out = None
        self._register_gradient_hook()

    def forward(self, input):
        """Performs the forward pass and caches the output sequence length."""
        output = super().forward(input)
        # Cache the output length (L_out) for use in the gradient hook.
        self.L_out = output.size(2)
        return output

    def _register_gradient_hook(self):
        """
        Registers a backward hook to scale gradients.

        The hook divides the gradients by the output sequence length (L_out).
        This normalization prevents gradients from scaling with the input length,
        leading to more stable training on variable-sized data.
        """
        def hook_fn(grad):
            if self.L_out is not None and self.L_out > 0:
                # Normalize the gradient by the number of filter applications.
                return grad / self.L_out
            return grad

        # The hook is applied to the gradients of the weight and bias parameters.
        self.weight.register_hook(hook_fn)
        if self.bias is not None:
            self.bias.register_hook(hook_fn)

class OpNormConv2d(nn.Conv2d):
    """
    A 2D convolutional layer that normalizes gradients by spatial operations.

    This layer extends `torch.nn.Conv2d` and scales the gradients of its weights
    and biases during the backward pass. The scaling factor is the product of
    the output height and width (H_out * W_out), which corresponds to the
    number of times the convolutional filter is applied. This helps stabilize
    training for models that process variable-size images.

    Since this class inherits from `nn.Conv2d`, it can be used as a drop-in
    replacement.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.H_out = None
        self.W_out = None
        self._register_gradient_hook()

    def forward(self, input):
        """Performs the forward pass and caches the output dimensions."""
        output = super().forward(input)
        # Cache the output spatial dimensions for use in the gradient hook.
        self.H_out, self.W_out = output.size(2), output.size(3)
        return output

    def _register_gradient_hook(self):
        """
        Registers a backward hook to scale gradients.

        The hook divides the gradients by the number of spatial locations
        (H_out * W_out). This normalization prevents gradients from scaling
        with input size, leading to more stable training.
        """
        def hook_fn(grad):
            if self.H_out is not None and self.W_out is not None:
                scaling_factor = self.H_out * self.W_out
                if scaling_factor > 0:
                    # Normalize the gradient by the number of filter applications.
                    return grad / scaling_factor
            return grad

        # The hook is applied to the gradients of the weight and bias parameters.
        self.weight.register_hook(hook_fn)
        if self.bias is not None:
            self.bias.register_hook(hook_fn) 
