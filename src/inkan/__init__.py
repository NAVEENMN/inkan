"""InKAN — Fast uniform cubic B-spline KAN layers via stable piecewise polynomial.

Evaluates B-spline basis functions via a symmetric piecewise polynomial
that avoids the catastrophic cancellation of the alternating truncated-power
sum. ``torch.compile`` can fuse the elementwise basis computation on
supported backends; set ``compile_basis=False`` for eager-mode execution
(required for higher-order autograd in PINN applications).

Algebraically equivalent to Cox-de Boor recursion for uniform cubic
splines, with guaranteed non-negative output. Partition of unity holds
on the configured grid range, up to floating-point error.

Currently supports cubic splines only (spline_order=3).

Example:
    >>> import torch
    >>> from inkan import KANLayer
    >>> layer = KANLayer(784, 64)
    >>> x = torch.randn(32, 784)
    >>> y = layer(x)  # [32, 64]
"""

__version__ = "0.5.0"

from inkan.layer import KANLayer
from inkan.network import KANNetwork
from inkan.basis import bspline_basis_eager, bspline_basis_local_eager
from inkan.visualize import plot_basis, plot_activations, plot_network, plot_surface

__all__ = ["KANLayer", "KANNetwork",
           "plot_basis", "plot_activations", "plot_network", "plot_surface"]
