"""InKAN in 20 lines — the simplest possible example."""

import torch
from inkan import KANLayer, KANNetwork

# Single layer (drop-in for nn.Linear)
layer = KANLayer(8, 4)
x = torch.randn(32, 8)
y = layer(x)
print(f"KANLayer: {x.shape} -> {y.shape}")

# Multi-layer network
net = KANNetwork([784, 64, 10])
x = torch.randn(32, 784)
y = net(x)
print(f"KANNetwork: {x.shape} -> {y.shape}")

# Training works with standard PyTorch
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss = y.sum()
loss.backward()
optimizer.step()
print("Backward + optimizer step: OK")
print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")
