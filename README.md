# InKAN

InKAN provies a fast and accurate B-spline [Kolmogorov-Arnold Network](https://arxiv.org/abs/2404.19756) layers for PyTorch.

**6-15x faster** than standard Cox-de Boor implementations (PyKAN, efficient-kan), **faster than Gaussian RBF** alternatives (FastKAN), while producing **exact B-spline basis values** with compact support, C2 continuity, and partition of unity.

## How it works

Standard KAN implementations compute B-spline basis functions using the Cox-de Boor recursion: 3 sequential passes for cubic splines, each creating intermediate tensors. InKAN replaces this with the truncated power closed form:

```
N(u) = (1/6) [relu(u)³ - 4·relu(u-1)³ + 6·relu(u-2)³ - 4·relu(u-3)³ + relu(u-4)³]
```

This single expression computes exact B-spline values with no recursion, no span lookups, and no gather operations. `torch.compile` fuses all elementwise ops into one GPU kernel.

## Installation

```bash
pip install inkan
```

**Requirements:** Python >= 3.9, PyTorch >= 2.0

**Supported devices:** CPU, CUDA (NVIDIA), MPS (Apple Silicon)

### From source

```bash
git clone https://github.com/NAVEENMN/inkan.git
cd inkan
pip install -e .
```

## Quick start

```python
import torch
from inkan import KANLayer, KANNetwork

# Drop-in replacement for nn.Linear
layer = KANLayer(784, 64)
x = torch.randn(32, 784)
y = layer(x)  # [32, 64]

# Multi-layer network
net = KANNetwork([784, 64, 10])
y = net(torch.randn(32, 784))  # [32, 10]
```

### MNIST example

```python
import torch
import torch.nn as nn
from inkan import KANNetwork

model = KANNetwork([784, 64, 10])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Standard PyTorch training loop
for images, labels in train_loader:
    output = model(images.view(-1, 784))
    loss = criterion(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

See [`examples/`](examples/) for complete runnable scripts.

## Visualization

InKAN includes built-in visualization for learned activation functions, similar to PyKAN's `model.plot()`.

```python
from inkan import KANNetwork, plot_basis, plot_activations, plot_network

model = KANNetwork([784, 32, 10], grid_size=8)
# ... train on MNIST ...

plot_basis(model.layers[0])          # B-spline basis bumps
plot_activations(model.layers[1])    # learned curves per edge
plot_network(model)                  # full network diagram
```

### B-spline basis functions

The 8 basis bumps (grid_size=5, degree=3), compact support, smooth overlap:

![Basis functions](https://raw.githubusercontent.com/NAVEENMN/inkan/main/assets/basis.png)

### Learned activation functions

After training on MNIST, each edge learns a unique activation curve.
Cyan = total, red dashed = spline component, green dotted = SiLU base:

![Learned activations](https://raw.githubusercontent.com/NAVEENMN/inkan/main/assets/activations.png)

### Network diagram

Full [784 → 32 → 10] network with learned curves on edges:

![Network diagram](https://raw.githubusercontent.com/NAVEENMN/inkan/main/assets/network.png)

## API

### `KANLayer(in_features, out_features, grid_size=5, spline_order=3)`

A single KAN layer. Drop-in replacement for `nn.Linear`.

| Parameter | Default | Description |
|---|---|---|
| `in_features` | — | Input dimension |
| `out_features` | — | Output dimension |
| `grid_size` | 5 | Number of knot intervals (more = finer approximation) |
| `spline_order` | 3 | B-spline degree (3 = cubic, recommended) |
| `grid_range` | (-1, 1) | Input range for the spline grid |

### `KANNetwork(layer_dims, grid_size=5, spline_order=3)`

Stack of KAN layers.

```python
# 3-layer KAN: 784 -> 128 -> 64 -> 10
net = KANNetwork([784, 128, 64, 10])
```

## Benchmarks

Forward pass time (ms) on Apple M-series GPU (MPS), batch=256:

| Layer | MNIST (784→64) | FashionMNIST (784→64) | CIFAR-10 (3072→64) |
|---|---|---|---|
| **InKAN (compiled)** | **0.27** | **0.20** | **0.39** |
| FastKAN (Gaussian RBF) | 0.38 | 0.31 | 0.99 |
| NoGather (unrolled) | 0.99 | 1.02 | 4.85 |
| Vanilla (Cox-de Boor) | 1.69 | 1.67 | 5.98 |

InKAN is **6.3x faster** than vanilla Cox-de Boor on MNIST and **15.3x faster** on CIFAR-10.

### Why it's fast

91% of a standard KAN forward pass is spent computing B-spline basis functions. InKAN eliminates this bottleneck:

| Approach | Basis cost | Why |
|---|---|---|
| Cox-de Boor (PyKAN) | 3 sequential GPU passes | Each pass depends on previous |
| Gaussian RBF (FastKAN) | 1 `exp()` call | Fast but not a true B-spline |
| **Truncated power (InKAN)** | **1 fused kernel** | `clamp + multiply` is cheaper than `exp()` |

## B-spline properties preserved

Unlike Gaussian RBF approximations, InKAN computes **exact** B-spline basis values:

- **Compact support**: each basis function is exactly zero outside its knot span window
- **C2 continuity**: second derivatives are continuous at every knot
- **Partition of unity**: basis values sum to 1 at every point in the interior
- **Non-negativity**: all basis values are >= 0

Verified: max difference vs Cox-de Boor reference is < 5e-5 in float32.

## Project structure

```
src/inkan/
├── __init__.py      # Public API
├── basis.py         # Truncated power B-spline + torch.compile (core math)
├── layer.py         # KANLayer
├── network.py       # KANNetwork
└── visualize.py     # plot_basis, plot_activations, plot_network
```

5 source files. The core innovation is in `basis.py`: 30 lines of math.

## Citation

If you use InKAN in your research, please cite:

```bibtex
@software{inkan2026,
  title={InKAN: B-Spline KANs via Truncated Power Form},
  author={Mysore, Naveen},
  year={2026},
  url={https://github.com/NAVEENMN/inkan}
}
```

## License

MIT
