"""InKAN — Fast B-spline KAN layers using the truncated power basis.

Computes exact B-spline basis functions via a closed-form formula
that torch.compile fuses into a single GPU kernel. 6-15x faster
than Cox-de Boor recursion, faster than Gaussian RBF alternatives,
while maintaining true B-spline properties (compact support, C2
continuity, partition of unity).

Example:
    >>> import torch
    >>> from inkan import KANLayer
    >>> layer = KANLayer(784, 64)
    >>> x = torch.randn(32, 784)
    >>> y = layer(x)  # [32, 64]
"""

__version__ = "0.2.0"

from inkan.layer import KANLayer
from inkan.network import KANNetwork
from inkan.visualize import plot_basis, plot_activations, plot_network

__all__ = ["KANLayer", "KANNetwork",
           "plot_basis", "plot_activations", "plot_network"]
