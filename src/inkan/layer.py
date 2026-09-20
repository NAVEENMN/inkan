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
import torch.nn.functional as F

from inkan.basis import (bspline_basis, bspline_basis_eager,
                         bspline_basis_local, bspline_basis_local_eager)


class KANLayer(nn.Module):
    """Kolmogorov-Arnold Network layer with fast B-spline activations.

    Uses a stable piecewise polynomial for basis computation.
    Produces exact B-spline values with compact support, C2 continuity,
    and partition of unity on the configured grid range (up to
    floating-point error).

    Args:
        in_features: Size of each input sample. For dim=2, must be 2.
        out_features: Size of each output sample.
        grid_size: Number of interior knot intervals. Default: 5.
        spline_order: Degree of the B-spline. Default: 3 (cubic).
        dim: Spline dimension. 1 = univariate (default), 2 = tensor-product surface.
        base_activation: Residual activation function. Default: SiLU.
        grid_range: Range of the input grid. Default: (-1, 1).
        compile_basis: If True (default), use torch.compile for basis
            computation. Set to False for eager-mode execution, which
            supports higher-order autograd (e.g. double backward for
            PDE/PINN applications).

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
        compile_basis: bool = True,
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
        if compile_basis:
            self._basis_fn = bspline_basis_local
        else:
            self._basis_fn = bspline_basis_local_eager

        n_bases = grid_size + spline_order
        self.n_bases = n_bases
        h = (grid_range[1] - grid_range[0]) / grid_size
        self.inv_h = 1.0 / h

        grid_starts = (torch.arange(n_bases, dtype=torch.get_default_dtype()) * h
                       + grid_range[0] - spline_order * h)
        self.register_buffer("grid_starts", grid_starts)

        if dim == 1:
            # Packed weight: [O, I*K + I] = [spline | base] in one parameter.
            # One F.linear call replaces two separate matmuls + addition.
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features * (n_bases + 1)))
        else:
            # 2D: coefficient matrix C per output: [out, K, K]
            # Not packed — the 3-operand tensor-product contraction
            # doesn't reduce to a single matmul.
            self._spline_weight = nn.Parameter(
                torch.empty(out_features, n_bases, n_bases))
            self._base_weight = nn.Parameter(
                torch.empty(out_features, 2))

        self.base_activation = base_activation()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.spline_weight, std=0.1)
        nn.init.trunc_normal_(self.base_weight, std=0.1)

    @property
    def spline_weight(self) -> torch.Tensor:
        """Spline coefficients. For 1D: view of shape [O, I, K] into packed weight."""
        if self.dim == 1:
            IK = self.in_features * self.n_bases
            return self.weight[:, :IK].reshape(
                self.out_features, self.in_features, self.n_bases)
        return self._spline_weight

    @property
    def base_weight(self) -> torch.Tensor:
        """Residual weights. For 1D: view of shape [O, I] into packed weight."""
        if self.dim == 1:
            IK = self.in_features * self.n_bases
            return self.weight[:, IK:]
        return self._base_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            return self._forward_1d(x)
        else:
            return self._forward_2d(x)

    def _forward_1d(self, x: torch.Tensor) -> torch.Tensor:
        """1D forward: each edge has a univariate B-spline activation.

        Concatenates spline basis features and residual activation into
        one feature vector [B, I*(K+1)], then contracts with the packed
        weight [O, I*(K+1)] in a single F.linear call.
        """
        batch = x.shape[0]
        bases = self._basis_fn(x, self.grid_starts, self.inv_h, self.n_bases)
        # [B, I*K | I] = [spline features | residual features]
        features = torch.cat([bases.reshape(batch, -1),
                              self.base_activation(x)], dim=1)
        return F.linear(features, self.weight)

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """2D forward: tensor-product surface S(x,y) = b_x^T C b_y."""
        # Compute bases for both coordinates in one call
        bases = self._basis_fn(x, self.grid_starts, self.inv_h, self.n_bases)  # [B, 2, K]
        b_x = bases[:, 0, :]  # [B, K]
        b_y = bases[:, 1, :]  # [B, K]

        # S(x,y) = b_x^T C b_y per output
        # Contract right first: [O, K, K] @ [B, K, 1] -> [O, B, K] via matmul
        # Then dot with b_x. einsum is clearest here and matches BLAS for 3-operand.
        spline_out = torch.einsum("bi,oij,bj->bo", b_x, self.spline_weight, b_y)

        # Residual: base_activation on both coordinates
        base_act = self.base_activation(x)
        base_out = F.linear(base_act, self.base_weight)

        return spline_out + base_out

    def extra_repr(self) -> str:
        parts = [f"in_features={self.in_features}",
                 f"out_features={self.out_features}",
                 f"grid_size={self.grid_size}",
                 f"spline_order={self.spline_order}"]
        if self.dim == 2:
            parts.append(f"dim={self.dim}")
        return ", ".join(parts)
