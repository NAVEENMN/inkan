"""B-spline basis computation via the truncated power closed form.

The cubic B-spline basis function can be expressed exactly as:

    N(u) = (1/6) * [relu(u)^3 - 4*relu(u-1)^3 + 6*relu(u-2)^3
                    - 4*relu(u-3)^3 + relu(u-4)^3]

where u = (x - grid_start_i) / h.

This eliminates Cox-de Boor recursion (3 sequential passes),
span lookups, and scatter/gather operations. torch.compile fuses
all elementwise ops into a single GPU kernel.

Two evaluation modes:

- **bounded** (default, ``bounded=True``): clamps u to [0, 4] before
  the alternating sum. This prevents the catastrophic cancellation
  that occurs when large u^3 terms cancel to zero in float32
  arithmetic. Best for KAN training in float32.

- **unbounded** (``bounded=False``): promotes to float64 and uses
  relu(u) without upper clamping. This preserves C2 smoothness at
  the support boundary so analytic first and second derivatives are
  consistent with the value basis. Required for PDE parameter
  recovery and physics-informed applications.

For uniform grids, both modes produce numerically equivalent results
to Cox-de Boor (verified max diff < 5e-5 in float32 at grid_size=5).
"""

import torch


def _bspline_basis(x: torch.Tensor, grid_starts: torch.Tensor,
                   inv_h: float, bounded: bool = True) -> torch.Tensor:
    """Compute B-spline basis values for all inputs and all basis functions.

    Args:
        x: Input tensor [batch, in_features].
        grid_starts: Start position of each basis function's support [n_bases].
        inv_h: Reciprocal of knot spacing (1/h).
        bounded: If True (default), clamp u to [0, 4] before evaluation.
            This prevents float32 cancellation far from the support but
            makes the basis non-smooth at the support boundary, breaking
            analytic derivatives.  Set to False for derivative-consistent
            evaluation (recommended with float64).

    Returns:
        Basis values [batch, in_features, n_bases].
    """
    if not bounded:
        # Unbounded mode: promote to float64 BEFORE computing u so the
        # subtraction x - grid_starts and all subsequent arithmetic
        # happen in double precision.  The alternating sum cancels
        # exactly for u >= 4 by the binomial identity, but this
        # cancellation requires float64 to avoid O(eps * u^3) errors.
        x = x.double()
        grid_starts = grid_starts.double()

    # u[b, i, j] = (x[b,i] - grid_starts[j]) / h
    u = (x.unsqueeze(-1) - grid_starts) * inv_h

    if bounded:
        # Bounded-coordinate stabilization: clamp u to the support [0, 4].
        # The B-spline is exactly zero for u <= 0 and u >= 4, so clamping
        # does not change any in-support value. But it bounds all intermediate
        # terms, preventing the O(eps * u^3) cancellation error that grows
        # with distance from the support.
        u = torch.clamp(u, 0.0, 4.0)
        r0 = u
    else:
        r0 = torch.clamp(u, min=0.0)

    # Truncated power form: sum of (u-k)_+^3 with binomial coefficients
    # Coefficients: C(4,k) * (-1)^k / 3! = [1, -4, 6, -4, 1] / 6
    r1 = torch.clamp(u - 1.0, min=0.0)
    r2 = torch.clamp(u - 2.0, min=0.0)
    r3 = torch.clamp(u - 3.0, min=0.0)
    r4 = torch.clamp(u - 4.0, min=0.0)

    return (r0*r0*r0 - 4.0*r1*r1*r1 + 6.0*r2*r2*r2
            - 4.0*r3*r3*r3 + r4*r4*r4) * 0.16666667


# torch.compile fuses all elementwise ops into a single GPU kernel.
# Works on CUDA, MPS, and CPU.
bspline_basis = torch.compile(_bspline_basis)

# Export the unfused version for testing/debugging and for PINNs
# (torch.compile does not support double backward).
bspline_basis_eager = _bspline_basis
