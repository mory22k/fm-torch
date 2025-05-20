# fm-torch

[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![PyTorch Version](https://img.shields.io/badge/torch-2.7.0%2B-orange)](https://pytorch.org/)

A PyTorch implementation of Factorization Machines (FM) with custom autograd function for efficient training.

## Overview

Factorization Machines (FM) are a class of models that capture feature interactions using factorized parameters. This implementation offers:

- Efficient computation of second-order feature interactions
- Custom PyTorch autograd function for optimized backward pass
- Simple API similar to standard PyTorch modules

The model is defined as:

$$\hat{y}(x) = b + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle \boldsymbol{v}_i, \boldsymbol{v}_j \rangle x_i x_j$$

Where:
- $b$ is the bias term
- $w_i$ are the weights of the linear terms
- $\boldsymbol{v}_i$ are k-dimensional factorized vectors
- $\langle \cdot, \cdot \rangle$ denotes the dot product

This implementation includes a custom `FactorizationMachineFunction` that efficiently computes both the forward pass and the gradients for backpropagation. The second-order interaction term is calculated using the formula:

$$\sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle \boldsymbol{v}_i, \boldsymbol{v}_j \rangle x_i x_j = \frac{1}{2} \left( \left\| \sum_{i=1}^{n} x_i \boldsymbol{v}_i \right\|^2 - \sum_{i=1}^{n} x_i^2 \|\boldsymbol{v}_i\|^2 \right)$$

This reduces the computational complexity from $O(n^2k)$ to $O(nk)$.

## References

1. S. Rendle, Factorization Machines, in *2010 IEEE International Conference on Data Mining* (IEEE, 2010), pp. 995–1000.

## Quick Start

```python
import torch
from fm_torch import SecondOrderFactorizationMachine

# Initialize model
dim_input = 10   # Input feature dimension
dim_factors = 8  # Latent factor dimension
model = SecondOrderFactorizationMachine(dim_input, dim_factors)

# Forward pass
batch_size = 32
x = torch.randn(batch_size, dim_input)
y_pred = model(x)  # Shape: (batch_size,)

# Training
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
y_true = torch.randn(batch_size)  # Replace with actual labels
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y_true)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/fm-torch.git
cd fm-torch

# If you are using mise, trust and install the dependencies
mise trust
mise install

# Set up development environment
uv sync
```

### Development Tools

This project uses:
- `mise` for development environment management
- `task` for running common development tasks
- `uv` for python package management
- `ruff` for linting and formatting
- `mypy` for type checking

```bash
# Format code
task format

# Check code style
task check

# Fix autofixable issues
task fix

# Prepare and commit
task commit:prepare:src
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
