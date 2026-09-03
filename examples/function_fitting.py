"""Fit a KAN to a known function — KAN's home turf.

Demonstrates that KAN can learn mathematical functions with
fewer parameters than an MLP, especially compositional ones.

Usage:
    pip install inkan matplotlib
    python examples/function_fitting.py
"""

import torch
import torch.nn as nn
import torch.optim as optim

from inkan import KANNetwork


# Target functions to fit
FUNCTIONS = {
    "sin": {
        "fn": lambda x: torch.sin(x[:, 0:1] + x[:, 1:2]),
        "in_dim": 2,
        "desc": "sin(x0 + x1)",
    },
    "product": {
        "fn": lambda x: (x[:, 0:1] * x[:, 1:2]),
        "in_dim": 2,
        "desc": "x0 * x1",
    },
    "composite": {
        "fn": lambda x: torch.sin(x[:, 0:1]) * torch.exp(-x[:, 1:2] ** 2),
        "in_dim": 2,
        "desc": "sin(x0) * exp(-x1^2)",
    },
}


def fit_function(name, epochs=1000, hidden=16, grid_size=10, lr=1e-3):
    cfg = FUNCTIONS[name]
    fn = cfg["fn"]
    in_dim = cfg["in_dim"]
    print(f"\nFitting: {cfg['desc']}")

    # Generate data
    torch.manual_seed(42)
    X_train = torch.randn(2048, in_dim)
    Y_train = fn(X_train)
    X_test = torch.randn(512, in_dim)
    Y_test = fn(X_test)

    # Build model
    model = KANNetwork([in_dim, hidden, 1], grid_size=grid_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        pred = model(X_train)
        loss = criterion(pred, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_loss = criterion(model(X_test), Y_test).item()
            print(f"  Epoch {epoch:>4d}  train_mse={loss.item():.6f}  "
                  f"test_mse={test_loss:.6f}")

    # Final
    model.eval()
    with torch.no_grad():
        final_mse = criterion(model(X_test), Y_test).item()
    print(f"  Final test MSE: {final_mse:.6f}")
    return final_mse


def main():
    print("InKAN Function Fitting")
    print("=" * 50)

    results = {}
    for name in FUNCTIONS:
        mse = fit_function(name)
        results[name] = mse

    print("\n" + "=" * 50)
    print("Summary:")
    for name, mse in results.items():
        print(f"  {FUNCTIONS[name]['desc']:<25s}  MSE = {mse:.6f}")


if __name__ == "__main__":
    main()
