# LLMs are the first authors of this file
# I do review them tho.

import pytest
import torch
import torch.nn as nn
from opnorm_grad.layers import OpNormConv1d, OpNormConv2d

# A list of configurations to test both 1D and 2D convolutional layers
CONV_CONFIGS = [
    {
        "opnorm_layer": OpNormConv1d,
        "torch_layer": nn.Conv1d,
        "input_shape": (2, 4, 16),  # (N, C_in, L)
        "layer_args": {"in_channels": 4, "out_channels": 8, "kernel_size": 3},
    },
    {
        "opnorm_layer": OpNormConv2d,
        "torch_layer": nn.Conv2d,
        "input_shape": (2, 3, 16, 16),  # (N, C_in, H, W)
        "layer_args": {"in_channels": 3, "out_channels": 6, "kernel_size": 3},
    },
]

@pytest.fixture(params=CONV_CONFIGS)
def conv_setup(request):
    """
    Provides a parameterized test setup for both 1D and 2D conv layers.

    This fixture creates a standard torch layer and an OpNorm layer with
    identical weights by copying them in-place. This method ensures that the
    gradient hooks on the OpNorm layer are preserved for testing.
    """
    config = request.param
    torch.manual_seed(42)

    # Create the standard and OpNorm layers
    std_conv = config["torch_layer"](**config["layer_args"])
    opnorm_conv = config["opnorm_layer"](**config["layer_args"])

    # Copy weights in-place to preserve the gradient hook on the OpNorm layer
    # Since OpNorm layers now inherit directly, we access weight and bias directly
    opnorm_conv.weight.data.copy_(std_conv.weight.data)
    opnorm_conv.bias.data.copy_(std_conv.bias.data)

    dummy_input = torch.randn(*config["input_shape"])
    
    return std_conv, opnorm_conv, dummy_input

def test_forward_pass_correctness(conv_setup):
    """
    Ensures the forward pass of OpNorm layers matches standard torch layers.
    
    This confirms the wrapper doesn't alter the layer's output, which is a
    pre-requisite for correct gradient scaling.
    """
    std_conv, opnorm_conv, dummy_input = conv_setup
    
    std_output = std_conv(dummy_input)
    opnorm_output = opnorm_conv(dummy_input)

    assert opnorm_output.shape == std_output.shape, "Output shape mismatch."
    assert torch.allclose(opnorm_output, std_output), "Forward pass output mismatch."

def test_gradient_scaling(conv_setup):
    """
    Verifies that gradients are correctly scaled by the number of operations.

    Compares the OpNorm gradient with a manually scaled standard gradient to
    confirm the hook works correctly.
    """
    std_conv, opnorm_conv, dummy_input = conv_setup

    # --- Standard Conv Gradient ---
    std_output = std_conv(dummy_input)
    std_loss = std_output.sum()
    std_loss.backward()

    # --- OpNorm Conv Gradient ---
    opnorm_output = opnorm_conv(dummy_input)
    opnorm_loss = opnorm_output.sum()
    opnorm_loss.backward()

    # Determine the scaling factor (L_out for 1D, H_out * W_out for 2D)
    if isinstance(opnorm_conv, OpNormConv1d):
        scaling_factor = opnorm_conv.L_out
        assert scaling_factor == std_output.shape[2], "Incorrect scaling factor for Conv1d."
    elif isinstance(opnorm_conv, OpNormConv2d):
        scaling_factor = opnorm_conv.H_out * opnorm_conv.W_out
        assert scaling_factor == std_output.shape[2] * std_output.shape[3], "Incorrect scaling factor for Conv2d."

    # Compare gradients - now accessing weight and bias directly
    scaled_std_grad = std_conv.weight.grad / scaling_factor
    assert torch.allclose(opnorm_conv.weight.grad, scaled_std_grad), "Weight gradient scaling is incorrect."
    
    if std_conv.bias is not None:
        scaled_std_bias_grad = std_conv.bias.grad / scaling_factor
        assert torch.allclose(opnorm_conv.bias.grad, scaled_std_bias_grad), "Bias gradient scaling is incorrect."

def test_weight_bias_assignment(conv_setup):
    """
    Tests if weights and biases can be correctly assigned to OpNorm layers.

    This confirms that the custom layers' weight and bias setters work as
    expected, allowing them to be drop-in replacements for standard layers
    where parameter reassignment might occur.
    """
    _, opnorm_conv, _ = conv_setup

    # Create new weight and bias tensors with the correct shapes
    new_weight = torch.randn_like(opnorm_conv.weight)
    new_bias = torch.randn_like(opnorm_conv.bias)

    # Assign them to the OpNorm layer as nn.Parameter
    opnorm_conv.weight = nn.Parameter(new_weight)
    opnorm_conv.bias = nn.Parameter(new_bias)

    # Check if the assignment was successful by comparing the underlying data
    assert torch.equal(opnorm_conv.weight.data, new_weight)
    assert torch.equal(opnorm_conv.bias.data, new_bias)
