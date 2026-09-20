"""Visualization for InKAN layers and networks.

All plot functions accept either a KANLayer or a KANNetwork.
When a network is passed, use the ``layer`` parameter to select
which layer to visualize (default: 0).

Example:
    >>> from inkan import KANNetwork
    >>> from inkan.visualize import plot_activations, plot_basis
    >>> net = KANNetwork([784, 32, 10])
    >>> # After training...
    >>> plot_activations(net, layer=0)   # first layer
    >>> plot_activations(net, layer=1)   # second layer
    >>> plot_basis(net)                  # defaults to layer 0
    >>> plot_surface(net, layer=0)       # 2D surface (dim=2 layer)
    >>> plot_network(net)                # full network diagram
"""

import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from inkan.basis import bspline_basis_eager


def _check_matplotlib():
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def _resolve_layer(model_or_layer, layer=None):
    """Resolve a KANLayer from either a layer or a network + index.

    Args:
        model_or_layer: A KANLayer or KANNetwork instance.
        layer: Layer index when a KANNetwork is passed. Default: 0.

    Returns:
        A KANLayer instance.
    """
    from inkan.layer import KANLayer
    from inkan.network import KANNetwork

    if isinstance(model_or_layer, KANLayer):
        if layer is not None:
            raise ValueError(
                "layer parameter is not used when passing a KANLayer directly")
        return model_or_layer

    if isinstance(model_or_layer, KANNetwork):
        idx = layer if layer is not None else 0
        n = len(model_or_layer.layers)
        if idx < 0 or idx >= n:
            raise IndexError(
                f"layer={idx} is out of range for a network with {n} layers")
        return model_or_layer.layers[idx]

    raise TypeError(
        f"Expected KANLayer or KANNetwork, got {type(model_or_layer).__name__}")


def plot_basis(model_or_layer, layer=None, n_points=500, figsize=None,
               title=None):
    """Plot the B-spline basis functions for a layer.

    Shows the individual basis bumps N_i(x) across the input range.

    Args:
        model_or_layer: A KANLayer or KANNetwork instance.
        layer: Layer index when a KANNetwork is passed. Default: 0.
        n_points: Number of sample points for the curves.
        figsize: Figure size tuple. Default auto-scales.
        title: Custom title.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()
    kan_layer = _resolve_layer(model_or_layer, layer)

    grid_starts = kan_layer.grid_starts.cpu()
    inv_h = kan_layer.inv_h
    h = 1.0 / inv_h
    n_bases = len(grid_starts)

    x_min = grid_starts[0].item()
    x_max = grid_starts[-1].item() + 4 * h
    x = torch.linspace(x_min, x_max, n_points)

    bases = bspline_basis_eager(x.unsqueeze(0), grid_starts, inv_h)
    bases = bases.squeeze(0).detach().numpy()
    x_np = x.numpy()

    if figsize is None:
        figsize = (10, 4)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, n_bases))

    for i in range(n_bases):
        ax.plot(x_np, bases[:, i], color=colors[i], linewidth=2,
                label=f"N{i}", alpha=0.8)
        peak_idx = np.argmax(bases[:, i])
        ax.plot(x_np[peak_idx], bases[peak_idx, i], 'o',
                color=colors[i], markersize=4)

    ax.axvline(x=-1.0, color='gray', linestyle='--', alpha=0.3,
               label='grid range')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.3)

    knot_positions = grid_starts.numpy() + 2 * h
    for kp in knot_positions:
        ax.axvline(x=kp, color='lightgray', linestyle=':', alpha=0.2)

    ax.set_xlabel("x")
    ax.set_ylabel("N(x)")
    ax.set_title(title or f"B-spline Basis Functions "
                 f"(grid_size={kan_layer.grid_size}, "
                 f"degree={kan_layer.spline_order})")
    ax.legend(fontsize=8, ncol=min(n_bases, 4), loc='upper right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def plot_surface(model_or_layer, layer=None, n_points=50, out_idx=0,
                 figsize=None, title=None):
    """Plot the learned 2D tensor-product B-spline surface.

    Shows S(x,y) = b_x^T C b_y as a 3D surface plot and a contour plot.

    Args:
        model_or_layer: A KANLayer (dim=2) or KANNetwork instance.
        layer: Layer index when a KANNetwork is passed. Default: 0.
        n_points: Grid resolution per axis.
        out_idx: Which output dimension to plot. Default: 0.
        figsize: Figure size tuple.
        title: Custom title.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D

    kan_layer = _resolve_layer(model_or_layer, layer)

    if kan_layer.dim != 2:
        raise ValueError("plot_surface requires a dim=2 KANLayer")

    grid_starts = kan_layer.grid_starts.cpu()
    inv_h = kan_layer.inv_h
    C = kan_layer.spline_weight[out_idx].detach().cpu()
    base_w = kan_layer.base_weight[out_idx].detach().cpu()

    x_lin = torch.linspace(-1, 1, n_points)
    y_lin = torch.linspace(-1, 1, n_points)
    xx, yy = torch.meshgrid(x_lin, y_lin, indexing="ij")

    b_x_vals = bspline_basis_eager(x_lin.unsqueeze(1), grid_starts, inv_h)
    b_x_vals = b_x_vals.squeeze(1)

    b_y_vals = bspline_basis_eager(y_lin.unsqueeze(1), grid_starts, inv_h)
    b_y_vals = b_y_vals.squeeze(1)

    zz_spline = torch.einsum("ik,kl,jl->ij", b_x_vals, C, b_y_vals)

    silu = torch.nn.functional.silu
    base_x = silu(x_lin) * base_w[0]
    base_y = silu(y_lin) * base_w[1]
    zz_base = base_x.unsqueeze(1) + base_y.unsqueeze(0)

    zz = (zz_spline + zz_base).detach().numpy()
    xx_np = xx.numpy()
    yy_np = yy.numpy()

    if figsize is None:
        figsize = (14, 5)

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(xx_np, yy_np, zz, cmap="viridis", alpha=0.9,
                     edgecolor="none")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_title(title or f"Learned 2D Surface (output {out_idx})")

    ax2 = fig.add_subplot(1, 2, 2)
    c = ax2.contourf(xx_np, yy_np, zz, levels=30, cmap="viridis")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")
    ax2.set_title("Contour View")
    plt.colorbar(c, ax=ax2)

    plt.tight_layout()
    return fig


def plot_activations(model_or_layer, layer=None, n_points=500,
                     in_idx=None, out_idx=None, figsize=None, title=None):
    """Plot the learned activation functions for each edge.

    Each subplot shows one edge's activation: the weighted sum of
    basis functions plus the SiLU residual.

    Args:
        model_or_layer: A KANLayer or KANNetwork instance.
        layer: Layer index when a KANNetwork is passed. Default: 0.
        n_points: Number of sample points.
        in_idx: Which input features to show. Default: all (capped at 8).
        out_idx: Which output features to show. Default: all (capped at 8).
        figsize: Figure size.
        title: Custom title.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()
    kan_layer = _resolve_layer(model_or_layer, layer)

    max_show = 8
    if in_idx is None:
        in_idx = list(range(min(kan_layer.in_features, max_show)))
    if out_idx is None:
        out_idx = list(range(min(kan_layer.out_features, max_show)))

    n_in = len(in_idx)
    n_out = len(out_idx)

    grid_starts = kan_layer.grid_starts.cpu()
    inv_h = kan_layer.inv_h
    spline_w = kan_layer.spline_weight.detach().cpu()
    base_w = kan_layer.base_weight.detach().cpu()

    x = torch.linspace(-1.0, 1.0, n_points)
    bases = bspline_basis_eager(x.unsqueeze(0), grid_starts, inv_h)
    bases = bases.squeeze(0)
    silu_x = torch.nn.functional.silu(x)

    if figsize is None:
        figsize = (2.5 * n_in, 2.2 * n_out)

    fig, axes = plt.subplots(n_out, n_in, figsize=figsize, squeeze=False)

    for row, oi in enumerate(out_idx):
        for col, ii in enumerate(in_idx):
            ax = axes[row, col]

            w = spline_w[oi, ii, :]
            spline_y = (bases * w).sum(dim=-1).numpy()

            bw = base_w[oi, ii].item()
            base_y = (silu_x * bw).numpy()

            total_y = spline_y + base_y

            x_np = x.numpy()
            ax.plot(x_np, total_y, color='#00c8ff', linewidth=1.5,
                    label='total')
            ax.plot(x_np, spline_y, color='#ff6464', linewidth=1.0,
                    alpha=0.6, linestyle='--', label='spline')
            ax.plot(x_np, base_y, color='#88cc88', linewidth=1.0,
                    alpha=0.6, linestyle=':', label='base')

            ax.set_title(f"({ii}\u2192{oi})", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.15)

            if row == 0 and col == 0:
                ax.legend(fontsize=6)

    layer_label = ""
    if isinstance(model_or_layer, _get_network_class()):
        idx = layer if layer is not None else 0
        layer_label = f" (layer {idx})"

    fig.suptitle(title or f"Learned Activation Functions{layer_label}",
                 fontsize=12)
    plt.tight_layout()
    return fig


def plot_network(model, x=None, max_nodes=16, figsize=None, title=None):
    """Plot the full KAN network with learned activation curves on edges.

    Args:
        model: A KANNetwork instance.
        x: Optional sample input (unused, reserved for future).
        max_nodes: Max nodes to draw per layer. Default: 16.
        figsize: Figure size.
        title: Custom title.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()

    layers = model.layers
    n_layers = len(layers)
    dims = [layers[0].in_features] + [l.out_features for l in layers]

    display_dims = [min(d, max_nodes) for d in dims]
    max_display = max(display_dims)

    if figsize is None:
        figsize = (4 * n_layers + 2, max_display * 0.6 + 2)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(-0.5, n_layers + 0.5)
    ax.set_ylim(-1.2, max_display - 0.2)
    ax.set_aspect('equal')
    ax.axis('off')

    node_x = []
    node_y = []
    for layer_idx in range(n_layers + 1):
        dim = display_dims[layer_idx]
        offset = (max_display - dim) / 2.0
        xs = [layer_idx] * dim
        ys = [offset + i for i in range(dim)]
        node_x.append(xs)
        node_y.append(ys)

    n_curve_pts = 50
    t_curve = torch.linspace(-1, 1, n_curve_pts)

    for li, layer in enumerate(layers):
        if layer.dim != 1:
            continue

        grid_starts = layer.grid_starts.cpu()
        inv_h = layer.inv_h
        spline_w = layer.spline_weight.detach().cpu()
        base_w = layer.base_weight.detach().cpu()

        bases = bspline_basis_eager(t_curve.unsqueeze(0), grid_starts, inv_h)
        bases = bases.squeeze(0)
        silu_t = torch.nn.functional.silu(t_curve)

        d_in = display_dims[li]
        d_out = display_dims[li + 1]

        for ii in range(d_in):
            for oi in range(d_out):
                x0, y0 = node_x[li][ii], node_y[li][ii]
                x1, y1 = node_x[li + 1][oi], node_y[li + 1][oi]

                w = spline_w[oi, ii, :]
                activation = (bases * w).sum(dim=-1) + silu_t * base_w[oi, ii]
                act_np = activation.detach().numpy()

                act_range = act_np.max() - act_np.min()
                if act_range > 1e-6:
                    act_norm = (act_np - act_np.min()) / act_range - 0.5
                else:
                    act_norm = np.zeros_like(act_np)

                strength = min(1.0, act_range / 2.0)

                t_edge = np.linspace(0, 1, n_curve_pts)
                cx = x0 + (x1 - x0) * t_edge
                cy = y0 + (y1 - y0) * t_edge

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
                        linewidth=0.5 + strength,
                        alpha=0.3 + strength * 0.5)

    for layer_idx in range(n_layers + 1):
        d = display_dims[layer_idx]
        for i in range(d):
            circle = plt.Circle(
                (node_x[layer_idx][i], node_y[layer_idx][i]),
                0.15, fill=True, color='#333333', ec='#00c8ff',
                linewidth=1.5, zorder=5
            )
            ax.add_patch(circle)
            ax.text(node_x[layer_idx][i], node_y[layer_idx][i],
                    str(i), ha='center', va='center', fontsize=7,
                    color='white', zorder=6)

    for li in range(n_layers + 1):
        real = dims[li]
        shown = display_dims[li]
        extra = f"\n(showing {shown}/{real})" if real > shown else ""
        label = f"Layer {li}\n({real}){extra}"
        ax.text(li, -1.0, label, ha='center', fontsize=9, color='gray')

    ax.set_title(title or "KAN Network", fontsize=13, pad=10)
    plt.tight_layout()
    return fig


def _get_network_class():
    """Lazy import to avoid circular dependency."""
    from inkan.network import KANNetwork
    return KANNetwork
