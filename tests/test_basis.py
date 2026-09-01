"""Tests for B-spline basis computation correctness."""

import torch
import pytest

from flashkan.basis import bspline_basis_eager


def _cox_de_boor_basis(i, k, x, grid):
    """Reference Cox-de Boor recursion (scalar, for verification)."""
    if k == 0:
        return 1.0 if grid[i] <= x < grid[i + 1] else 0.0
    d1 = grid[i + k] - grid[i]
    d2 = grid[i + k + 1] - grid[i + 1]
    left = 0.0
    if d1 != 0:
        left = (x - grid[i]) / d1 * _cox_de_boor_basis(i, k - 1, x, grid)
    right = 0.0
    if d2 != 0:
        right = (grid[i + k + 1] - x) / d2 * _cox_de_boor_basis(i + 1, k - 1, x, grid)
    return left + right


def _reference_bases(x_vals, grid_size=5, spline_order=3):
    """Compute basis values using Cox-de Boor for a batch of scalars."""
    h = 2.0 / grid_size
    n_bases = grid_size + spline_order
    grid = [(i - spline_order) * h - 1.0
            for i in range(n_bases + spline_order + 1)]
    result = torch.zeros(len(x_vals), n_bases)
    for xi, x in enumerate(x_vals):
        for bi in range(n_bases):
            result[xi, bi] = _cox_de_boor_basis(bi, spline_order, x, grid)
    return result


class TestBasisCorrectness:
    """Verify truncated power basis matches Cox-de Boor."""

    def test_matches_cox_de_boor(self):
        grid_size, spline_order = 5, 3
        n_bases = grid_size + spline_order
        h = 2.0 / grid_size
        inv_h = 1.0 / h
        grid_starts = torch.arange(n_bases).float() * h - 1.0 - spline_order * h

        x_vals = torch.linspace(-0.99, 0.99, 100)
        x = x_vals.unsqueeze(0)  # [1, 100]

        bases = bspline_basis_eager(x, grid_starts, inv_h)
        ref = _reference_bases(x_vals.tolist(), grid_size, spline_order)

        assert bases.shape == (1, 100, n_bases)
        assert torch.allclose(bases.squeeze(0), ref, atol=1e-3)

    def test_partition_of_unity(self):
        """Basis functions should sum to 1 in the interior."""
        grid_size, spline_order = 5, 3
        n_bases = grid_size + spline_order
        h = 2.0 / grid_size
        grid_starts = torch.arange(n_bases).float() * h - 1.0 - spline_order * h

        x = torch.linspace(-0.9, 0.9, 200).unsqueeze(0)
        bases = bspline_basis_eager(x, grid_starts, 1.0 / h)
        sums = bases.sum(dim=-1)

        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3)

    def test_compact_support(self):
        """Each basis should be nonzero only over ~4 knot spans."""
        grid_size, spline_order = 10, 3
        n_bases = grid_size + spline_order
        h = 2.0 / grid_size
        grid_starts = torch.arange(n_bases).float() * h - 1.0 - spline_order * h

        x = torch.tensor([[0.0]])  # single point
        bases = bspline_basis_eager(x, grid_starts, 1.0 / h)
        nonzero = (bases.abs() > 1e-6).sum().item()
        assert nonzero <= spline_order + 1  # at most degree+1 active

    def test_non_negative(self):
        """B-spline basis values should be non-negative."""
        grid_size, spline_order = 5, 3
        n_bases = grid_size + spline_order
        h = 2.0 / grid_size
        grid_starts = torch.arange(n_bases).float() * h - 1.0 - spline_order * h

        x = torch.randn(64, 32).clamp(-1, 1)
        bases = bspline_basis_eager(x, grid_starts, 1.0 / h)
        assert (bases >= -1e-4).all()  # small float32 rounding errors OK

    def test_c2_smoothness(self):
        """Second derivative should be continuous (no jumps)."""
        grid_size, spline_order = 5, 3
        n_bases = grid_size + spline_order
        h = 2.0 / grid_size
        grid_starts = torch.arange(n_bases).float() * h - 1.0 - spline_order * h

        # Fine grid for numerical derivatives
        x = torch.linspace(-0.9, 0.9, 10000).unsqueeze(0)
        bases = bspline_basis_eager(x, grid_starts, 1.0 / h)

        # Numerical second derivative
        d2 = bases[:, 2:] - 2 * bases[:, 1:-1] + bases[:, :-2]
        # Check for jumps: max change in second derivative
        d2_diff = (d2[:, 1:] - d2[:, :-1]).abs().max()
        assert d2_diff < 0.01  # smooth, no discontinuities


class TestBasisShapes:
    """Verify output shapes for various inputs."""

    @pytest.mark.parametrize("batch,in_f,grid_size", [
        (1, 1, 5), (32, 64, 5), (256, 784, 10), (4, 8, 20),
    ])
    def test_output_shape(self, batch, in_f, grid_size):
        spline_order = 3
        n_bases = grid_size + spline_order
        h = 2.0 / grid_size
        grid_starts = torch.arange(n_bases).float() * h - 1.0 - spline_order * h

        x = torch.randn(batch, in_f)
        bases = bspline_basis_eager(x, grid_starts, 1.0 / h)
        assert bases.shape == (batch, in_f, n_bases)


class TestBasisGradients:
    """Verify gradients flow correctly."""

    def test_backward(self):
        grid_size, spline_order = 5, 3
        n_bases = grid_size + spline_order
        h = 2.0 / grid_size
        grid_starts = torch.arange(n_bases).float() * h - 1.0 - spline_order * h

        x = torch.randn(4, 8, requires_grad=True)
        bases = bspline_basis_eager(x, grid_starts, 1.0 / h)
        bases.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_finite(self):
        grid_size, spline_order = 5, 3
        n_bases = grid_size + spline_order
        h = 2.0 / grid_size
        grid_starts = torch.arange(n_bases).float() * h - 1.0 - spline_order * h

        x = torch.randn(4, 8, requires_grad=True)
        bases = bspline_basis_eager(x, grid_starts, 1.0 / h)
        bases.sum().backward()
        assert torch.isfinite(x.grad).all()
