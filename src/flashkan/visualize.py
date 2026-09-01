"""Visualization for FlashKAN layers and networks.

Plot learned activation functions, basis functions, and full
network diagrams. All plots use matplotlib.

Example:
    >>> from flashkan import KANLayer
    >>> from flashkan.visualize import plot_activations, plot_basis
    >>> layer = KANLayer(2, 3, grid_size=5)
    >>> # After training...
    >>> plot_activations(layer)            # learned curves per edge
    >>> plot_basis(layer)                  # B-spline basis bumps
    >>> plot_network(model, x_sample)      # full network diagram
"""

import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from flashkan.basis import bspline_basis_eager


def _check_matplotlib():
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install flashkan[dev]"
        )


def plot_basis(layer, n_points=500, figsize=None, title=None):
    """Plot the B-spline basis functions for a layer.

    Shows the individual basis bumps N_i(x) across the input range.
    This is what each basis function looks like before being weighted
    by the learned coefficients.

    Args:
        layer: A KANLayer instance.
        n_points: Number of sample points for the curves.
        figsize: Figure size tuple. Default auto-scales.
        title: Custom title.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()

    grid_starts = layer.grid_starts.cpu()
    inv_h = layer.inv_h
    h = 1.0 / inv_h
    n_bases = len(grid_starts)

    # Input range: cover the full support of all basis functions
    x_min = grid_starts[0].item()
    x_max = grid_starts[-1].item() + 4 * h
    x = torch.linspace(x_min, x_max, n_points)

    # Evaluate all basis functions
    # basis expects [batch, in], we use [1, n_points]
    bases = bspline_basis_eager(x.unsqueeze(0), grid_starts, inv_h)
    bases = bases.squeeze(0).detach().numpy()  # [n_points, n_bases]
    x_np = x.numpy()

    if figsize is None:
        figsize = (10, 4)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, n_bases))

    for i in range(n_bases):
        ax.plot(x_np, bases[:, i], color=colors[i], linewidth=2,
                label=f"N{i}", alpha=0.8)
        # Mark the center of each basis
        peak_idx = np.argmax(bases[:, i])
        ax.plot(x_np[peak_idx], bases[peak_idx, i], 'o',
                color=colors[i], markersize=4)

    # Mark the active grid range
    grid_range = layer.grid_starts.cpu()
    ax.axvline(x=-1.0, color='gray', linestyle='--', alpha=0.3, label='grid range')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.3)

    # Mark knot positions
    knot_positions = grid_starts.numpy() + 2 * h  # center of support
    for kp in knot_positions:
        ax.axvline(x=kp, color='lightgray', linestyle=':', alpha=0.2)

    ax.set_xlabel("x")
    ax.set_ylabel("N(x)")
    ax.set_title(title or f"B-spline Basis Functions (grid_size={layer.grid_size}, degree={layer.spline_order})")
    ax.legend(fontsize=8, ncol=min(n_bases, 4), loc='upper right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def plot_activations(layer, n_points=500, in_idx=None, out_idx=None,
                     figsize=None, title=None):
    """Plot the learned activation functions φ(x) for each edge.

    Each subplot shows one edge's activation: the weighted sum of
    basis functions plus the SiLU residual.

    Args:
        layer: A KANLayer instance.
        n_points: Number of sample points.
        in_idx: Which input features to show. Default: all (capped at 8).
        out_idx: Which output features to show. Default: all (capped at 8).
        figsize: Figure size.
        title: Custom title.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()

    # Determine which edges to plot
    max_show = 8
    if in_idx is None:
        in_idx = list(range(min(layer.in_features, max_show)))
    if out_idx is None:
        out_idx = list(range(min(layer.out_features, max_show)))

    n_in = len(in_idx)
    n_out = len(out_idx)

    grid_starts = layer.grid_starts.cpu()
    inv_h = layer.inv_h
    spline_w = layer.spline_weight.detach().cpu()  # [out, in, n_bases]
    base_w = layer.base_weight.detach().cpu()       # [out, in]

    # Sample x values
    x = torch.linspace(-1.0, 1.0, n_points)
    bases = bspline_basis_eager(x.unsqueeze(0), grid_starts, inv_h)
    bases = bases.squeeze(0)  # [n_points, n_bases]
    silu_x = torch.nn.functional.silu(x)

    if figsize is None:
        figsize = (2.5 * n_in, 2.2 * n_out)

    fig, axes = plt.subplots(n_out, n_in, figsize=figsize, squeeze=False)

    for row, oi in enumerate(out_idx):
        for col, ii in enumerate(in_idx):
            ax = axes[row, col]

            # Spline component: bases @ spline_w[oi, ii, :]
            w = spline_w[oi, ii, :]  # [n_bases]
            spline_y = (bases * w).sum(dim=-1).numpy()

            # Base component: silu(x) * base_w[oi, ii]
            bw = base_w[oi, ii].item()
            base_y = (silu_x * bw).numpy()

            # Total activation
            total_y = spline_y + base_y

            x_np = x.numpy()
            ax.plot(x_np, total_y, color='#00c8ff', linewidth=1.5,
                    label='total')
            ax.plot(x_np, spline_y, color='#ff6464', linewidth=1.0,
                    alpha=0.6, linestyle='--', label='spline')
            ax.plot(x_np, base_y, color='#88cc88', linewidth=1.0,
                    alpha=0.6, linestyle=':', label='base')

            ax.set_title(f"({ii}→{oi})", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.15)

            if row == 0 and col == 0:
                ax.legend(fontsize=6)

    fig.suptitle(title or "Learned Activation Functions", fontsize=12)
    plt.tight_layout()
    return fig


def plot_network(model, x=None, figsize=None, title=None):
    """Plot the full KAN network with learned activation curves on edges.

    Args:
        model: A KANNetwork instance.
        x: Optional sample input to show activation magnitudes.
        figsize: Figure size.
        title: Custom title.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()

    layers = model.layers
    n_layers = len(layers)
    dims = [layers[0].in_features] + [l.out_features for l in layers]

    if figsize is None:
        figsize = (4 * n_layers + 2, max(dims) * 0.6 + 1)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(-0.5, n_layers + 0.5)
    ax.set_ylim(-0.5, max(dims) - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Node positions
    node_x = []
    node_y = []
    for layer_idx in range(n_layers + 1):
        dim = dims[layer_idx]
        offset = (max(dims) - dim) / 2.0
        xs = [layer_idx] * dim
        ys = [offset + i for i in range(dim)]
        node_x.append(xs)
        node_y.append(ys)

    # Draw edges with mini activation curves
    n_curve_pts = 50
    t_curve = torch.linspace(-1, 1, n_curve_pts)

    for li, layer in enumerate(layers):
        grid_starts = layer.grid_starts.cpu()
        inv_h = layer.inv_h
        spline_w = layer.spline_weight.detach().cpu()
        base_w = layer.base_weight.detach().cpu()

        bases = bspline_basis_eager(t_curve.unsqueeze(0), grid_starts, inv_h)
        bases = bases.squeeze(0)
        silu_t = torch.nn.functional.silu(t_curve)

        in_dim = layer.in_features
        out_dim = layer.out_features

        for ii in range(in_dim):
            for oi in range(out_dim):
                x0, y0 = node_x[li][ii], node_y[li][ii]
                x1, y1 = node_x[li + 1][oi], node_y[li + 1][oi]

                # Compute activation curve
                w = spline_w[oi, ii, :]
                activation = (bases * w).sum(dim=-1) + silu_t * base_w[oi, ii]
                act_np = activation.detach().numpy()

                # Normalize curve height for display
                act_range = act_np.max() - act_np.min()
                if act_range > 1e-6:
                    act_norm = (act_np - act_np.min()) / act_range - 0.5
                else:
                    act_norm = np.zeros_like(act_np)

                # Edge strength (for color intensity)
                strength = min(1.0, act_range / 2.0)

                # Draw the curve along the edge
                t_edge = np.linspace(0, 1, n_curve_pts)
                cx = x0 + (x1 - x0) * t_edge
                cy = y0 + (y1 - y0) * t_edge

                # Perpendicular offset for the curve
                dx, dy = x1 - x0, y1 - y0
                length = (dx**2 + dy**2) ** 0.5
                if length > 0:
                    nx, ny = -dy / length, dx / length
                else:
                    nx, ny = 0, 1

                curve_scale = 0.15
                cx_curved = cx + nx * act_norm * curve_scale
                cy_curved = cy + ny * act_norm * curve_scale

                color = plt.cm.coolwarm(0.5 + strength * 0.5)
                ax.plot(cx_curved, cy_curved, color=color,
                        linewidth=0.5 + strength, alpha=0.3 + strength * 0.5)

    # Draw nodes
    for layer_idx in range(n_layers + 1):
        dim = dims[layer_idx]
        for i in range(dim):
            circle = plt.Circle(
                (node_x[layer_idx][i], node_y[layer_idx][i]),
                0.15, fill=True, color='#333333', ec='#00c8ff',
                linewidth=1.5, zorder=5
            )
            ax.add_patch(circle)
            ax.text(node_x[layer_idx][i], node_y[layer_idx][i],
                    str(i), ha='center', va='center', fontsize=7,
                    color='white', zorder=6)

    # Layer labels
    for li in range(n_layers + 1):
        label = f"Layer {li}\n({dims[li]})"
        ax.text(li, -0.8, label, ha='center', fontsize=9, color='gray')

    ax.set_title(title or "KAN Network", fontsize=13, pad=10)
    plt.tight_layout()
    return fig
