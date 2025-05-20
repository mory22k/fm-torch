"""Factorization Machine using PyTorch."""

from typing import Any, Tuple

import torch
from torch import nn


class FactorizationMachineFunction(torch.autograd.Function):
    """Autograd Function for fast second-order Factorization Machine."""

    @staticmethod
    def forward(  # type: ignore
        ctx,
        inputs: torch.Tensor,
        bias: torch.Tensor,
        weights: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the FM output.

        Args:
            inputs (torch.Tensor): Input tensor with shape (batch_size, dim_input).
            bias (torch.Tensor): Bias term (scalar).
            weights (torch.Tensor): Linear weights with shape (dim_input,).
            V (torch.Tensor): Factor matrix with shape (dim_input, num_factors).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size,).
        """
        q = inputs @ V  # (B, k)
        linear_term = inputs @ weights  # (B,)

        square_sum = (q**2).sum(dim=1)  # (B,)
        sum_square = (inputs**2 @ V**2).sum(dim=1)  # (B,)
        second_order = 0.5 * (square_sum - sum_square)  # (B,)
        output = bias + linear_term + second_order  # (B,)

        ctx.save_for_backward(inputs, q, weights, V)
        return output

    @staticmethod
    def backward(  # type: ignore
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gradients using accelerated formulas.

        Args:
            grad_output (torch.Tensor): Gradient of loss with shape (batch_size,).

        Returns:
            Tuple[None, torch.Tensor, torch.Tensor, torch.Tensor]:
                - grad_bias: scalar
                - grad_weights: (dim_input,)
                - grad_V: (dim_input, num_factors)
        """
        inputs: torch.Tensor = ctx.saved_tensors[0]
        q: torch.Tensor = ctx.saved_tensors[1]
        # weights: torch.Tensor = ctx.saved_tensors[2]
        V: torch.Tensor = ctx.saved_tensors[3]

        B: int = inputs.shape[0]

        grad_bias = grad_output.sum().unsqueeze(0)
        grad_weights = inputs.T @ grad_output

        xv = inputs.unsqueeze(2) * V.unsqueeze(0)
        term = q.unsqueeze(1) - xv
        grad_V = (grad_output.view(B, 1, 1) * term * inputs.unsqueeze(2)).sum(dim=0)

        return None, grad_bias, grad_weights, grad_V


class SecondOrderFactorizationMachine(nn.Module):
    """Second-order Factorization Machine model."""

    def __init__(self, dim_input: int, num_factors: int):
        """Initialize the model.

        Args:
            dim_input (int): Input dimension d.
            num_factors (int): Number of latent factors k.
        """
        super().__init__()
        self.linear_layer = nn.Linear(dim_input, 1, bias=True)
        self.V = nn.Parameter(torch.randn(dim_input, num_factors) * 0.01)

    @property
    def bias(self) -> torch.Tensor:
        """Bias term."""
        return self.linear_layer.bias

    @property
    def linear(self) -> torch.Tensor:
        """Linear weights."""
        return self.linear_layer.weight.view(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, dim_input).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size,).
        """
        output: torch.Tensor = FactorizationMachineFunction.apply(
            x, self.bias, self.linear, self.V
        )
        return output
