# InKAN

Fast, stable uniform cubic B-spline [Kolmogorov-Arnold Network](https://arxiv.org/abs/2404.19756) layers for PyTorch.

Evaluates B-spline basis functions via the truncated power closed form instead of the Cox-de Boor recursion. **2.8--3.5x lower forward-pass latency** than recursive implementations on H100 CUDA, numerically stable up to grid_size=200+ with bounded-coordinate evaluation that prevents the cancellation error historically associated with the truncated power form.

Supports 1D univariate splines and 2D tensor-product B-spline surfaces.

## How it works

Standard KAN implementations compute B-spline basis functions using the Cox-de Boor recursion: 3 sequential passes for cubic splines, each creating intermediate tensors. InKAN replaces this with the truncated power closed form:

```
N(u) = (1/6) [relu(u)³ - 4·relu(u-1)³ + 6·relu(u-2)³ - 4·relu(u-3)³ + relu(u-4)³]
```

Before evaluation, `u` is clamped to `[0, 4]` (bounded-coordinate stabilization). This is algebraically correct (the B-spline is exactly zero outside its support) and prevents numerical cancellation on finer grids. `torch.compile` fuses all elementwise ops into one GPU kernel.

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

### 1D (default)

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

### 2D tensor-product surface

```python
from inkan import KANLayer

# Learns S(x,y) = b_x^T C b_y (no recursion)
layer = KANLayer(2, 1, dim=2, grid_size=12)
xy = torch.randn(32, 2)
z = layer(xy)  # [32, 1]

# Multi-output for parametric surfaces (R^2 -> R^3)
layer = KANLayer(2, 3, dim=2, grid_size=12)
xyz = layer(uv)  # [32, 3]
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

InKAN includes built-in visualization for learned activation functions and surfaces.

```python
from inkan import KANNetwork, plot_basis, plot_activations, plot_surface

# 1D: basis bumps and learned activations
model = KANNetwork([784, 32, 10], grid_size=8)
plot_basis(model.layers[0])
plot_activations(model.layers[1])

# 2D: learned surface
from inkan import KANLayer
layer = KANLayer(2, 1, dim=2, grid_size=12)
# ... train ...
plot_surface(layer)  # 3D surface + contour plot
```

### B-spline basis functions

The 8 basis bumps (grid_size=5, degree=3), compact support, smooth overlap:

![Basis functions](https://raw.githubusercontent.com/NAVEENMN/inkan/main/assets/basis.png)

### Learned activation functions

After training on MNIST, each edge learns a unique activation curve.
Cyan = total, red dashed = spline component, green dotted = SiLU base:

![Learned activations](https://raw.githubusercontent.com/NAVEENMN/inkan/main/assets/activations.png)

### 2D learned surface

Tensor-product B-spline surface fitting sin(pi*x)*sin(pi*y) with 227 parameters:

![2D surface](https://raw.githubusercontent.com/NAVEENMN/inkan/main/assets/surface_2d.png)

### Network diagram

Full [784 → 32 → 10] network with learned curves on edges:

![Network diagram](https://raw.githubusercontent.com/NAVEENMN/inkan/main/assets/network.png)

## API

### `KANLayer(in_features, out_features, grid_size=5, spline_order=3, dim=1, grid_range=(-1, 1))`

A single KAN layer. Drop-in replacement for `nn.Linear`.

| Parameter | Default | Description |
|---|---|---|
| `in_features` | -- | Input dimension (must be 2 for dim=2) |
| `out_features` | -- | Output dimension |
| `grid_size` | 5 | Number of knot intervals (more = finer approximation) |
| `spline_order` | 3 | B-spline degree (only 3 is currently supported) |
| `dim` | 1 | 1 = univariate spline per edge, 2 = tensor-product surface |
| `grid_range` | (-1, 1) | Input range for the spline grid |

### `KANNetwork(layer_dims, grid_size=5, spline_order=3, dim=1, grid_range=(-1, 1))`

Stack of KAN layers.

```python
# 1D: 3-layer KAN
net = KANNetwork([784, 128, 64, 10])

# 2D: first layer is tensor-product, rest are 1D
net = KANNetwork([2, 8, 1], dim=2, grid_size=12)
```

## Benchmarks

### Speed (H100 CUDA, forward pass, batch=256)

| Method | dim=784 | dim=3072 |
|---|---|---|
| **InKAN** | **0.253 ms** | **0.264 ms** |
| Efficient-KAN | 0.722 ms | 0.919 ms |
| FastKAN (Gaussian RBF) | 0.230 ms | 0.256 ms |

InKAN is **~3x faster** than Efficient-KAN. Comparable to FastKAN on H100.

### Numerical stability (bounded-coordinate stabilization)

| Grid G | Max out-of-support error | Partition-of-unity error |
|---|---|---|
| 5 | 0.00 | 2.2e-6 |
| 32 | 0.00 | 1.7e-6 |
| 64 | 0.00 | 1.2e-6 |
| 100 | 0.00 | 4.2e-6 |
| 200 | 0.00 | 8.6e-6 |

Without the clamp, grid_size=64 produces out-of-support errors of 0.04 and partition-of-unity errors of 0.25.

## B-spline properties preserved

Algebraically equivalent to Cox-de Boor for uniform cubic splines:

- **Compact support**: each basis function is exactly zero outside its knot span window (enforced by bounded-coordinate clamp)
- **C2 continuity**: second derivatives are continuous at every knot
- **Partition of unity**: basis values sum to 1 in the interior (error < 1e-5 at all tested grid sizes)
- **Non-negativity**: all basis values >= 0 within float32 tolerance

## Limitations

- **Cubic only**: currently supports spline_order=3. Other degrees are rejected with a clear error.
- **Uniform grids only**: non-uniform knot vectors are not supported. Adaptive grid refinement requires per-span coefficients.
- **dim=2 first layer only**: KANNetwork with dim=2 uses a tensor-product surface in the first layer; subsequent layers are 1D.

## Project structure

```
src/inkan/
├── __init__.py      # Public API
├── basis.py         # Truncated power B-spline + bounded-coordinate clamp + torch.compile
├── layer.py         # KANLayer (dim=1 and dim=2)
├── network.py       # KANNetwork
└── visualize.py     # plot_basis, plot_activations, plot_surface, plot_network
```

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
