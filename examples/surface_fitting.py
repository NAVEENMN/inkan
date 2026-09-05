"""Fit 2D surfaces using InKAN's tensor-product B-spline layer (dim=2).

Demonstrates:
- KANLayer with dim=2 for learning S(x,y) = b_x^T C b_y
- Fitting known 2D functions
- plot_surface() visualization

Usage:
    pip install inkan matplotlib
    python surface_fitting.py
"""

import torch
import torch.optim as optim
import math

from inkan import KANLayer, plot_surface


def main():
    functions = {
        "sin(pi*x)*sin(pi*y)": lambda xy: (
            torch.sin(math.pi * xy[:, 0:1]) * torch.sin(math.pi * xy[:, 1:2])
        ),
        "exp(-(x^2+y^2))": lambda xy: (
            torch.exp(-(xy[:, 0:1]**2 + xy[:, 1:2]**2))
        ),
        "x^2 - y^2": lambda xy: (
            xy[:, 0:1]**2 - xy[:, 1:2]**2
        ),
    }

    for name, fn in functions.items():
        print(f"\n--- Fitting: {name} ---")

        # 2D tensor-product spline layer
        layer = KANLayer(2, 1, dim=2, grid_size=12)
        n_params = sum(p.numel() for p in layer.parameters())
        print(f"  Params: {n_params}")

        # Training data: uniform in [-1, 1]^2
        xy = torch.rand(3000, 2) * 2 - 1
        z = fn(xy)

        optimizer = optim.Adam(layer.parameters(), lr=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

        for epoch in range(1, 2001):
            optimizer.zero_grad()
            loss = ((layer(xy) - z) ** 2).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 500 == 0:
                print(f"  Epoch {epoch:>4d}  MSE={loss.item():.2e}")

        # Visualize
        try:
            fig = plot_surface(layer, title=f"Learned: {name}")
            fname = name.replace("*", "").replace("(", "").replace(")", "")
            fname = fname.replace(" ", "_").replace("^", "")
            fig.savefig(f"surface_{fname}.png", dpi=150)
            print(f"  Saved: surface_{fname}.png")
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            print("  matplotlib not installed, skipping plot")


if __name__ == "__main__":
    main()
