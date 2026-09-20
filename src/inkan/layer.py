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

# Basis function lookup: (basis_mode, compile_basis) -> function
_BASIS_FNS = {
    ("dense", True): bspline_basis,
    ("dense", False): bspline_basis_eager,
    ("local", True): bspline_basis_local,
    ("local", False): bspline_basis_local_eager,
}


class KANLayer(nn.Module):
    """Kolmogorov-Arnold Network layer with fast B-spline activations.

    Uses a stable piecewise polynomial for basis computation.
    Produces exact B-spline values with compact support, C2 continuity,
    and partition of unity on the configured grid range (up to
    floating-point error).

    Both 1D and 2D layers pack spline and residual weights into a
    single parameter and use one F.linear call per forward pass.

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
        basis_mode: "local" (default) uses span lookup to evaluate only
            the 4 active bases per input. "dense" evaluates all K bases.
            Local is faster for large grids; dense may be preferable for
            small grids or when profiling shows scatter overhead dominates.

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
        basis_mode: str = "local",
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
        if basis_mode not in ("dense", "local"):
            raise ValueError(f"basis_mode must be 'dense' or 'local', got '{basis_mode}'")

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.dim = dim
        self.basis_mode = basis_mode
        self._basis_fn = _BASIS_FNS[(basis_mode, compile_basis)]

        n_bases = grid_size + spline_order
        self.n_bases = n_bases
        h = (grid_range[1] - grid_range[0]) / grid_size
        self.inv_h = 1.0 / h

        grid_starts = (torch.arange(n_bases, dtype=torch.get_default_dtype()) * h
                       + grid_range[0] - spline_order * h)
        self.register_buffer("grid_starts", grid_starts)

        # Packed weight: single parameter for both spline and residual.
        # 1D: [O, I*K + I]     features = [bases.flat | silu(x)]
        # 2D: [O, K*K + 2]     features = [outer_product.flat | silu(x), silu(y)]
        if dim == 1:
            n_spline_features = in_features * n_bases
        else:
            n_spline_features = n_bases * n_bases

        self.weight = nn.Parameter(
            torch.empty(out_features, n_spline_features + in_features))

        self.base_activation = base_activation()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.spline_weight, std=0.1)
        nn.init.trunc_normal_(self.base_weight, std=0.1)

    @property
    def spline_weight(self) -> torch.Tensor:
        """Spline coefficients as a view into the packed weight.

        Returns:
            1D: [O, I, K], 2D: [O, K, K]
        """
        if self.dim == 1:
            n = self.in_features * self.n_bases
            return self.weight[:, :n].reshape(
                self.out_features, self.in_features, self.n_bases)
        n = self.n_bases * self.n_bases
        return self.weight[:, :n].reshape(
            self.out_features, self.n_bases, self.n_bases)

    @property
    def base_weight(self) -> torch.Tensor:
        """Residual weights as a view into the packed weight.

        Returns:
            1D: [O, I], 2D: [O, 2]
        """
        if self.dim == 1:
            n = self.in_features * self.n_bases
        else:
            n = self.n_bases * self.n_bases
        return self.weight[:, n:]

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
        features = torch.cat([bases.reshape(batch, -1),
                              self.base_activation(x)], dim=1)
        return F.linear(features, self.weight)

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """2D forward: tensor-product surface via outer-product features.

        Computes outer product b_x (x) b_y^T, flattens to [B, K*K],
        concatenates with residual [B, 2], and contracts with packed
        weight [O, K*K+2] in a single F.linear call.
        """
        bases = self._basis_fn(x, self.grid_starts, self.inv_h, self.n_bases)
        b_x = bases[:, 0, :]  # [B, K]
        b_y = bases[:, 1, :]  # [B, K]

        # Tensor-product features: outer product flattened to [B, K*K]
        surface_features = (b_x.unsqueeze(-1) * b_y.unsqueeze(-2)).flatten(1)

        features = torch.cat([surface_features,
                              self.base_activation(x)], dim=1)
        return F.linear(features, self.weight)

    def extra_repr(self) -> str:
        parts = [f"in_features={self.in_features}",
                 f"out_features={self.out_features}",
                 f"grid_size={self.grid_size}",
                 f"spline_order={self.spline_order}"]
        if self.dim == 2:
            parts.append(f"dim={self.dim}")
        if self.basis_mode != "local":
            parts.append(f"basis_mode='{self.basis_mode}'")
        return ", ".join(parts)
