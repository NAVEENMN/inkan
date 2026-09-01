"""KANLayer — the core building block.

Drop-in replacement for nn.Linear in KAN networks:
    nn.Linear(in_features, out_features)
    KANLayer(in_features, out_features)

Each edge (i→j) has a learnable B-spline activation function
plus a SiLU residual connection, following the KAN paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashkan.basis import bspline_basis


class KANLayer(nn.Module):
    """Kolmogorov-Arnold Network layer with fast B-spline activations.

    Uses the truncated power closed form + torch.compile for basis
    computation. Produces exact B-spline values with compact support,
    C2 continuity, and partition of unity.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        grid_size: Number of interior knot intervals. More = finer
            approximation. Default: 5.
        spline_order: Degree of the B-spline. Default: 3 (cubic).
        base_activation: Residual activation function. Default: SiLU.
        grid_range: Range of the input grid. Default: (-1, 1).

    Shape:
        - Input: (batch, in_features)
        - Output: (batch, out_features)

    Example:
        >>> layer = KANLayer(784, 64, grid_size=10)
        >>> x = torch.randn(32, 784)
        >>> y = layer(x)  # [32, 64]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: type = nn.SiLU,
        grid_range: tuple = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        n_bases = grid_size + spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        self.inv_h = 1.0 / h

        # Grid start positions for each basis function
        grid_starts = (torch.arange(n_bases).float() * h
                       + grid_range[0] - spline_order * h)
        self.register_buffer("grid_starts", grid_starts)

        # Learnable spline coefficients: one per (output, input, basis)
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, n_bases)
        )

        # Learnable residual weights
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with scaled random values."""
        # Kaiming-style init scaled for spline basis
        nn.init.trunc_normal_(self.spline_weight, std=0.1)
        nn.init.trunc_normal_(self.base_weight, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, out_features).
        """
        # Spline activation: φ(x) = Σ_k c_k * N_k(x)
        bases = bspline_basis(x, self.grid_starts, self.inv_h)
        spline_out = torch.einsum("bin,oin->bo", bases, self.spline_weight)

        # Residual: b(x) * w
        base_out = torch.einsum(
            "bi,oi->bo", self.base_activation(x), self.base_weight
        )

        return spline_out + base_out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"grid_size={self.grid_size}, "
                f"spline_order={self.spline_order}")
