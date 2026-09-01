"""KANNetwork — multi-layer KAN with a simple API.

Example:
    >>> net = KANNetwork([784, 64, 10])
    >>> y = net(torch.randn(32, 784))  # [32, 10]
"""

import torch
import torch.nn as nn

from flashkan.layer import KANLayer


class KANNetwork(nn.Module):
    """Stack of KANLayers.

    Args:
        layer_dims: List of dimensions, e.g. [784, 64, 10] creates
            a 2-layer KAN: 784→64→10.
        grid_size: Grid size for all layers. Default: 5.
        spline_order: Spline degree for all layers. Default: 3.

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
    ):
        super().__init__()
        if len(layer_dims) < 2:
            raise ValueError("Need at least 2 dimensions (input + output)")

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(
                KANLayer(layer_dims[i], layer_dims[i + 1],
                         grid_size=grid_size, spline_order=spline_order)
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
