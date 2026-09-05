"""KANNetwork — multi-layer KAN with a simple API.

Example:
    >>> # 1D (default)
    >>> net = KANNetwork([784, 64, 10])
    >>> y = net(torch.randn(32, 784))  # [32, 10]
    >>>
    >>> # 2D tensor-product
    >>> net = KANNetwork([2, 8, 3], dim=2)
    >>> xyz = net(torch.randn(32, 2))  # [32, 3]
"""

import torch
import torch.nn as nn

from inkan.layer import KANLayer


class KANNetwork(nn.Module):
    """Stack of KANLayers.

    Args:
        layer_dims: List of dimensions, e.g. [784, 64, 10] creates
            a 2-layer KAN: 784->64->10.
        grid_size: Grid size for all layers. Default: 5.
        spline_order: Spline degree for all layers. Default: 3.
        dim: Spline dimension for all layers. 1 = univariate (default),
            2 = tensor-product surface (requires first dim to be 2).

    Example:
        >>> net = KANNetwork([784, 128, 64, 10])
        >>> x = torch.randn(32, 784)
        >>> y = net(x)  # [32, 10]
    """

    def __init__(
        self,
        layer_dims: list,
        grid_size: int = 5,
        spline_order: int = 3,
        dim: int = 1,
        grid_range: tuple = (-1.0, 1.0),
    ):
        super().__init__()
        if len(layer_dims) < 2:
            raise ValueError("Need at least 2 dimensions (input + output)")

        layers = []
        for i in range(len(layer_dims) - 1):
            # For dim=2, only the first layer takes 2D input;
            # subsequent layers are 1D over the intermediate features
            layer_dim = dim if (dim == 2 and i == 0) else 1
            layers.append(
                KANLayer(layer_dims[i], layer_dims[i + 1],
                         grid_size=grid_size, spline_order=spline_order,
                         dim=layer_dim, grid_range=grid_range)
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
