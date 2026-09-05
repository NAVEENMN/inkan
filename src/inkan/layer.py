"""KANLayer — the core building block.

Drop-in replacement for nn.Linear in KAN networks:
    nn.Linear(in_features, out_features)
    KANLayer(in_features, out_features)

Supports 1D univariate splines (default) and 2D tensor-product splines:
    KANLayer(784, 64)              # 1D: each edge has a univariate spline
    KANLayer(2, 1, dim=2)          # 2D: tensor-product surface S(x,y) = b_x^T C b_y
"""

import torch
import torch.nn as nn

from inkan.basis import bspline_basis


class KANLayer(nn.Module):
    """Kolmogorov-Arnold Network layer with fast B-spline activations.

    Uses the truncated power closed form + torch.compile for basis
    computation. Produces exact B-spline values with compact support,
    C2 continuity, and partition of unity.

    Args:
        in_features: Size of each input sample. For dim=2, must be 2.
        out_features: Size of each output sample.
        grid_size: Number of interior knot intervals. Default: 5.
        spline_order: Degree of the B-spline. Default: 3 (cubic).
        dim: Spline dimension. 1 = univariate (default), 2 = tensor-product surface.
        base_activation: Residual activation function. Default: SiLU.
        grid_range: Range of the input grid. Default: (-1, 1).

    Shape:
        - dim=1: Input (batch, in_features) -> Output (batch, out_features)
        - dim=2: Input (batch, 2) -> Output (batch, out_features)

    Example:
        >>> # 1D (default)
        >>> layer = KANLayer(784, 64, grid_size=10)
        >>> y = layer(torch.randn(32, 784))  # [32, 64]
        >>>
        >>> # 2D tensor-product surface
        >>> layer = KANLayer(2, 1, dim=2, grid_size=12)
        >>> z = layer(torch.randn(32, 2))  # [32, 1]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        dim: int = 1,
        base_activation: type = nn.SiLU,
        grid_range: tuple = (-1.0, 1.0),
    ):
        super().__init__()
        if dim not in (1, 2):
            raise ValueError(f"dim must be 1 or 2, got {dim}")
        if dim == 2 and in_features != 2:
            raise ValueError(f"dim=2 requires in_features=2, got {in_features}")
        if spline_order != 3:
            raise ValueError(
                f"Only cubic splines (spline_order=3) are currently supported. "
                f"Got spline_order={spline_order}. General-degree support is "
                f"planned for a future release."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.dim = dim

        n_bases = grid_size + spline_order
        self.n_bases = n_bases
        h = (grid_range[1] - grid_range[0]) / grid_size
        self.inv_h = 1.0 / h

        grid_starts = (torch.arange(n_bases).float() * h
                       + grid_range[0] - spline_order * h)
        self.register_buffer("grid_starts", grid_starts)

        if dim == 1:
            # 1D: one weight per (output, input, basis)
            self.spline_weight = nn.Parameter(
                torch.empty(out_features, in_features, n_bases))
            self.base_weight = nn.Parameter(
                torch.empty(out_features, in_features))
        else:
            # 2D: coefficient matrix C per output: [out, K, K]
            self.spline_weight = nn.Parameter(
                torch.empty(out_features, n_bases, n_bases))
            self.base_weight = nn.Parameter(
                torch.empty(out_features, 2))

        self.base_activation = base_activation()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.spline_weight, std=0.1)
        nn.init.trunc_normal_(self.base_weight, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            return self._forward_1d(x)
        else:
            return self._forward_2d(x)

    def _forward_1d(self, x: torch.Tensor) -> torch.Tensor:
        """1D forward: each edge has a univariate B-spline activation."""
        bases = bspline_basis(x, self.grid_starts, self.inv_h)
        spline_out = torch.einsum("bin,oin->bo", bases, self.spline_weight)
        base_out = torch.einsum(
            "bi,oi->bo", self.base_activation(x), self.base_weight)
        return spline_out + base_out

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """2D forward: tensor-product surface S(x,y) = b_x^T C b_y."""
        # Split input into x and y coordinates
        x_coord = x[:, 0:1]  # [batch, 1]
        y_coord = x[:, 1:2]  # [batch, 1]

        # Compute 1D bases independently
        b_x = bspline_basis(x_coord, self.grid_starts, self.inv_h)
        b_x = b_x.squeeze(1)  # [batch, K]

        b_y = bspline_basis(y_coord, self.grid_starts, self.inv_h)
        b_y = b_y.squeeze(1)  # [batch, K]

        # S(x,y) = b_x^T C b_y per output
        # spline_weight: [out, K, K], b_x: [batch, K], b_y: [batch, K]
        spline_out = torch.einsum("bi,oij,bj->bo",
                                  b_x, self.spline_weight, b_y)

        # Residual: base_activation on both coordinates
        base_act = self.base_activation(x)  # [batch, 2]
        base_out = torch.einsum("bi,oi->bo", base_act, self.base_weight)

        return spline_out + base_out

    def extra_repr(self) -> str:
        parts = [f"in_features={self.in_features}",
                 f"out_features={self.out_features}",
                 f"grid_size={self.grid_size}",
                 f"spline_order={self.spline_order}"]
        if self.dim == 2:
            parts.append(f"dim={self.dim}")
        return ", ".join(parts)
