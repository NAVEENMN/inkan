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
                         bspline_basis_local, bspline_basis_local_eager,
                         local_values)

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
        contraction: Contraction strategy for combining basis values
            with coefficients. Options:
            - "auto" (default): selects the fastest path per workload.
              Uses direct contraction for 2D (always) and for 1D when
              out_features <= 16 and n_bases >= 32. Uses dense F.linear
              otherwise.
            - "dense": always use dense feature matrix + F.linear.
            - "direct": always gather only active coefficients, skip
              dense expansion. Faster for narrow outputs and large grids,
              slower for wide outputs due to gather overhead.

    Shape:
        - dim=1: Input (batch, in_features) -> Output (batch, out_features)
        - dim=2: Input (batch, 2) -> Output (batch, out_features)

    Example:
        >>> layer = KANLayer(784, 64, grid_size=10)
        >>> y = layer(torch.randn(32, 784))  # [32, 64]
        >>>
        >>> # 2D: automatic direct contraction (19-235x faster)
        >>> layer = KANLayer(2, 1, dim=2, grid_size=50)
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
        contraction: str = "auto",
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
        if contraction not in ("auto", "dense", "direct"):
            raise ValueError(f"contraction must be 'auto', 'dense', or 'direct', got '{contraction}'")

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

        # Resolve contraction strategy
        if contraction == "auto":
            if dim == 2:
                self._use_direct = True
            else:
                self._use_direct = (out_features <= 16 and n_bases >= 32)
        else:
            self._use_direct = (contraction == "direct")

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
            if self._use_direct:
                return self._forward_1d_direct(x)
            return self._forward_1d_dense(x)
        else:
            if self._use_direct:
                return self._forward_2d_direct(x)
            return self._forward_2d_dense(x)

    def _forward_1d_dense(self, x: torch.Tensor) -> torch.Tensor:
        """1D forward via dense feature matrix + F.linear.

        Concatenates spline basis features and residual activation into
        one feature vector [B, I*K+I], then contracts with the packed
        weight [O, I*K+I] in a single F.linear call.
        """
        batch = x.shape[0]
        bases = self._basis_fn(x, self.grid_starts, self.inv_h, self.n_bases)
        features = torch.cat([bases.reshape(batch, -1),
                              self.base_activation(x)], dim=1)
        return F.linear(features, self.weight)

    def _forward_1d_direct(self, x: torch.Tensor) -> torch.Tensor:
        """1D forward via direct local contraction.

        Gathers only the 4 active coefficients per input feature,
        multiplies by basis values, and sums. Avoids the dense [B, I*K]
        intermediate. Faster when out_features is small and K is large.
        """
        indices, values = local_values(
            x, self.grid_starts, self.inv_h, self.n_bases)
        # indices: [B, I, 4], values: [B, I, 4]

        # Offset indices to index into the flattened I*K spline weight
        feature_offsets = torch.arange(
            self.in_features, device=x.device) * self.n_bases
        flat_indices = indices + feature_offsets[None, :, None]  # [B, I, 4]

        # Gather active coefficients: spline_weight is [O, I*K]
        # Reshape to [I*K, O] for F.embedding lookup
        n_spline = self.in_features * self.n_bases
        coeff_table = self.weight[:, :n_spline].T  # [I*K, O]
        coeffs = F.embedding(flat_indices, coeff_table)  # [B, I, 4, O]

        # Contract: sum over 4 active bases and I input features
        spline_out = (coeffs * values.unsqueeze(-1)).sum(dim=(1, 2))  # [B, O]

        # Add residual
        return spline_out + F.linear(self.base_activation(x), self.base_weight)

    def _forward_2d_dense(self, x: torch.Tensor) -> torch.Tensor:
        """2D forward via dense outer-product features + F.linear.

        Computes outer product b_x (x) b_y^T, flattens to [B, K*K],
        concatenates with residual [B, 2], and contracts with packed
        weight [O, K*K+2] in a single F.linear call.
        """
        bases = self._basis_fn(x, self.grid_starts, self.inv_h, self.n_bases)
        b_x = bases[:, 0, :]  # [B, K]
        b_y = bases[:, 1, :]  # [B, K]

        surface_features = (b_x.unsqueeze(-1) * b_y.unsqueeze(-2)).flatten(1)

        features = torch.cat([surface_features,
                              self.base_activation(x)], dim=1)
        return F.linear(features, self.weight)

    def _forward_2d_direct(self, x: torch.Tensor) -> torch.Tensor:
        """2D forward via direct 16-term local contraction.

        Evaluates 4 active x-bases and 4 active y-bases, forms 16
        tensor products, gathers the corresponding 16 coefficients,
        and sums. Avoids the dense [B, K*K] intermediate entirely.
        For grid=200 this is 235x faster than the dense path.
        """
        indices, values = local_values(
            x, self.grid_starts, self.inv_h, self.n_bases)
        # indices: [B, 2, 4], values: [B, 2, 4]

        ix = indices[:, 0, :]   # [B, 4] -- active x basis indices
        iy = indices[:, 1, :]   # [B, 4] -- active y basis indices
        vx = values[:, 0, :]    # [B, 4] -- active x basis values
        vy = values[:, 1, :]    # [B, 4] -- active y basis values

        # 16 flat indices into the K*K coefficient matrix
        flat_idx = (ix.unsqueeze(-1) * self.n_bases
                    + iy.unsqueeze(-2)).flatten(1)  # [B, 16]

        # 16 tensor-product basis values
        products = (vx.unsqueeze(-1) * vy.unsqueeze(-2)).flatten(1)  # [B, 16]

        # Gather active coefficients: [B, 16, O]
        n_spline = self.n_bases * self.n_bases
        coeff_table = self.weight[:, :n_spline].T  # [K*K, O]
        coeffs = F.embedding(flat_idx, coeff_table)  # [B, 16, O]

        # Contract: sum over 16 active terms
        spline_out = (coeffs * products.unsqueeze(-1)).sum(dim=1)  # [B, O]

        # Add residual
        return spline_out + F.linear(self.base_activation(x), self.base_weight)

    def extra_repr(self) -> str:
        parts = [f"in_features={self.in_features}",
                 f"out_features={self.out_features}",
                 f"grid_size={self.grid_size}",
                 f"spline_order={self.spline_order}"]
        if self.dim == 2:
            parts.append(f"dim={self.dim}")
        if self.basis_mode != "local":
            parts.append(f"basis_mode='{self.basis_mode}'")
        if self._use_direct:
            parts.append("contraction='direct'")
        return ", ".join(parts)
