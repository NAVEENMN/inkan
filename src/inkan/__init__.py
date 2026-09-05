"""InKAN — Fast uniform cubic B-spline KAN layers via truncated power basis.

Evaluates B-spline basis functions via the truncated power closed form
with bounded-coordinate stabilization. torch.compile fuses all
elementwise operations into a single GPU kernel.

Algebraically equivalent to Cox-de Boor recursion for uniform cubic
splines, with numerically equivalent outputs in float32 (verified
max deviation < 5e-5 at grid_size=5). Bounded-coordinate clamping
prevents cancellation on finer grids and out-of-support inputs.

Currently supports cubic splines only (spline_order=3).

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
from inkan.visualize import plot_basis, plot_activations, plot_network, plot_surface

__all__ = ["KANLayer", "KANNetwork",
           "plot_basis", "plot_activations", "plot_network", "plot_surface"]
