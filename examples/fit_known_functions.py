"""Fit InKAN to known 1D functions and visualize learned activations.

This script trains a tiny KAN (1 → hidden → 1) on sin, cos, x², etc.
and plots what each learned activation function looks like.

Usage:
    pip install inkan matplotlib
    python examples/fit_known_functions.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from inkan import KANNetwork
from inkan.basis import bspline_basis_eager

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    raise ImportError("matplotlib required: pip install matplotlib")


FUNCTIONS = {
    "sin(x)": torch.sin,
    "cos(x)": torch.cos,
    "x²": lambda x: x ** 2,
    "|x|": torch.abs,
    "exp(-x²)": lambda x: torch.exp(-x ** 2),
}


def train_model(fn, epochs=3000, hidden=1, grid_size=8, lr=1e-3):
    """Train a [1 → hidden → 1] KAN to fit a 1D function."""
    torch.manual_seed(42)
    x_train = torch.linspace(-1, 1, 1000).unsqueeze(1)
    y_train = fn(x_train)

    model = KANNetwork([1, hidden, 1], grid_size=grid_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        pred = model(x_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"    epoch {epoch:>4d}  mse={loss.item():.8f}")

    model.eval()
    with torch.no_grad():
        final_mse = criterion(model(x_train), y_train).item()
    return model, final_mse


def plot_learned_activations(layer, ax, title=""):
    """Plot the activation function learned by edge (0→0) of a layer."""
    grid_starts = layer.grid_starts.cpu()
    inv_h = layer.inv_h
    spline_w = layer.spline_weight.detach().cpu()  # [out, in, n_bases]
    base_w = layer.base_weight.detach().cpu()       # [out, in]

    x = torch.linspace(-1.0, 1.0, 500)
    bases = bspline_basis_eager(x.unsqueeze(0), grid_starts, inv_h)
    bases = bases.squeeze(0)  # [500, n_bases]
    silu_x = torch.nn.functional.silu(x)

    # Edge (0→0)
    w = spline_w[0, 0, :]
    spline_y = (bases * w).sum(dim=-1).numpy()
    bw = base_w[0, 0].item()
    base_y = (silu_x * bw).numpy()
    total_y = spline_y + base_y

    x_np = x.numpy()
    ax.plot(x_np, total_y, color='#00c8ff', linewidth=2.0, label='total φ(x)')
    ax.plot(x_np, spline_y, color='#ff6464', linewidth=1.2,
            alpha=0.7, linestyle='--', label='spline')
    ax.plot(x_np, base_y, color='#88cc88', linewidth=1.2,
            alpha=0.7, linestyle=':', label='base (SiLU)')
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.2)


def main():
    print("InKAN: Fitting Known Functions")
    print("=" * 50)

    models = {}
    for name, fn in FUNCTIONS.items():
        print(f"\n  Fitting {name}:")
        model, mse = train_model(fn, epochs=3000, hidden=1, grid_size=8)
        models[name] = (model, mse)
        print(f"    final mse = {mse:.8f}")

    # --- Figure 1: Fit quality (model output vs ground truth) ---
    n = len(FUNCTIONS)
    fig1, axes1 = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    x_test = torch.linspace(-1, 1, 500).unsqueeze(1)

    for idx, (name, fn) in enumerate(FUNCTIONS.items()):
        ax = axes1[idx]
        model, mse = models[name]
        with torch.no_grad():
            y_pred = model(x_test).squeeze().numpy()
        y_true = fn(x_test).squeeze().numpy()
        x_np = x_test.squeeze().numpy()

        ax.plot(x_np, y_true, 'k-', linewidth=2, label='true', alpha=0.7)
        ax.plot(x_np, y_pred, '--', color='#ff6464', linewidth=2, label='KAN')
        ax.set_title(f"{name}\nMSE={mse:.2e}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig1.suptitle("InKAN Fit: Model Output vs Ground Truth", fontsize=13)
    plt.tight_layout()
    fig1.savefig("fit_quality.png", dpi=150, bbox_inches='tight')
    print("\nSaved fit_quality.png")

    # --- Figure 2: Learned activation functions (layer 0, edge 0→0) ---
    fig2, axes2 = plt.subplots(1, n, figsize=(3.5 * n, 3.5))

    for idx, (name, fn) in enumerate(FUNCTIONS.items()):
        ax = axes2[idx]
        model, mse = models[name]
        layer = model.layers[0]
        plot_learned_activations(layer, ax, title=f"Layer 0: {name}")
        if idx == 0:
            ax.legend(fontsize=7)

    fig2.suptitle("Learned Activation Functions (Layer 0, edge 0→0)", fontsize=13)
    plt.tight_layout()
    fig2.savefig("learned_activations.png", dpi=150, bbox_inches='tight')
    print("Saved learned_activations.png")

    # --- Figure 3: Layer 1 activations ---
    fig3, axes3 = plt.subplots(1, n, figsize=(3.5 * n, 3.5))

    for idx, (name, fn) in enumerate(FUNCTIONS.items()):
        ax = axes3[idx]
        model, mse = models[name]
        layer = model.layers[1]
        plot_learned_activations(layer, ax, title=f"Layer 1: {name}")
        if idx == 0:
            ax.legend(fontsize=7)

    fig3.suptitle("Learned Activation Functions (Layer 1, edge 0→0)", fontsize=13)
    plt.tight_layout()
    fig3.savefig("learned_activations_layer1.png", dpi=150, bbox_inches='tight')
    print("Saved learned_activations_layer1.png")

    plt.show()


if __name__ == "__main__":
    main()
