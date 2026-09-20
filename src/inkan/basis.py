"""B-spline basis computation via stable piecewise polynomial.

Evaluates the cubic B-spline basis using a symmetric piecewise
polynomial that avoids the catastrophic cancellation of the
alternating truncated-power sum. ``torch.compile`` can fuse the
elementwise basis computation on supported backends (CUDA, MPS, CPU);
actual fusion depends on backend and configuration.

The cubic B-spline has support [0, 4] and is symmetric about u=2:

    Segment [0, 1]:  N(u) = u³ / 6
    Segment [1, 2]:  N(u) = (1 + 3v + 3v² - 3v³) / 6,  v = u - 1
    Segment [2, 3]:  mirror of [1, 2]
    Segment [3, 4]:  mirror of [0, 1]
    Outside:         N(u) = 0

Two evaluation modes:

- **dense** (``_bspline_basis``): evaluates all K basis functions for
  every input. Simple, branchless, good GPU utilization.

- **local** (``_bspline_basis_local``): finds the 4 active bases per
  input via floor() span lookup, evaluates only those 4, scatters into
  dense output. Faster when K is large; see benchmarks for crossover.

Algebraically equivalent to Cox-de Boor recursion for uniform cubic
splines. Output is guaranteed non-negative. Partition of unity holds
on the configured grid range, up to floating-point error.
"""

import torch


def _piecewise_eval(u: torch.Tensor) -> torch.Tensor:
    """Evaluate the cubic B-spline piecewise polynomial.

    Shared by both dense and local paths for consistent numerics.

    Args:
        u: Normalized coordinates (any shape). Values outside [0, 4]
           produce zero.

    Returns:
        Basis values (same shape as u).
    """
    s = torch.where(u <= 2.0, u, 4.0 - u).clamp_min(0.0)
    v = (s - 1.0).clamp_min(0.0)
    outer = s * s * s
    inner = 1.0 + v * (3.0 + v * (3.0 - 3.0 * v))
    return torch.where(s <= 1.0, outer, inner) / 6.0


def _bspline_basis(x: torch.Tensor, grid_starts: torch.Tensor,
                   inv_h: float, n_bases: int) -> torch.Tensor:
    """Compute B-spline basis values for all inputs and all basis functions.

    Dense evaluation: computes all K basis values per input.

    Args:
        x: Input tensor [batch, in_features].
        grid_starts: Start position of each basis function's support [n_bases].
        inv_h: Reciprocal of knot spacing (1/h).
        n_bases: Total number of basis functions (unused, for API compat).

    Returns:
        Basis values [batch, in_features, n_bases].
    """
    u = (x.unsqueeze(-1) - grid_starts) * inv_h
    return _piecewise_eval(u)


def _bspline_basis_local(x: torch.Tensor, grid_starts: torch.Tensor,
                         inv_h: float, n_bases: int) -> torch.Tensor:
    """Compute B-spline basis via local 4-basis evaluation.

    For each input, finds the 4 active basis functions using floor(),
    evaluates only those 4, and scatters into the full [B, I, K] output.

    Uses grid_starts[0] as the canonical origin for span lookup,
    consistent with the uniform grid construction.

    Args:
        x: Input tensor [batch, in_features].
        grid_starts: Start position of each basis function's support [n_bases].
        inv_h: Reciprocal of knot spacing (1/h).
        n_bases: Total number of basis functions (grid_size + spline_order).

    Returns:
        Basis values [batch, in_features, n_bases].
    """
    B, I = x.shape

    # Span lookup: find the 4 active bases per input.
    # t0 = normalized position relative to first grid_start
    t0 = (x - grid_starts[0]) * inv_h  # [B, I]
    j_start = (t0.floor().long() - 3).clamp(0, n_bases - 4)  # [B, I]

    # Compute u values for only the 4 active bases: [B, I, 4]
    offsets = torch.arange(4, device=x.device)
    idx = j_start.unsqueeze(-1) + offsets  # [B, I, 4]

    # Use consistent dtype: cast indices to input dtype (handles fp16/bf16)
    u = t0.unsqueeze(-1) - j_start.unsqueeze(-1).to(t0.dtype) - offsets.to(t0.dtype)

    # Evaluate piecewise polynomial on [B, I, 4] instead of [B, I, K]
    bases_local = _piecewise_eval(u)

    # Scatter into dense [B, I, K] for compatibility with F.linear contraction.
    # new_zeros inherits dtype and device from bases_local.
    bases = bases_local.new_zeros(B, I, n_bases)
    return bases.scatter(2, idx, bases_local)


# torch.compile can fuse the elementwise basis computation on
# supported backends (CUDA, MPS, CPU). Actual fusion depends on
# backend and torch version.
bspline_basis = torch.compile(_bspline_basis)
bspline_basis_local = torch.compile(_bspline_basis_local)

# Unfused versions for testing/debugging and for higher-order autograd
# (e.g. PINNs requiring double backward, which torch.compile does
# not support). Use compile_basis=False in KANLayer to select this.
bspline_basis_eager = _bspline_basis
bspline_basis_local_eager = _bspline_basis_local
