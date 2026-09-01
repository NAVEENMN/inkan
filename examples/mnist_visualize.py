"""Train MNIST and visualize what the KAN learned.

Produces:
  - results/mnist_basis.png        — B-spline basis functions
  - results/mnist_activations.png  — learned activation curves (layer 1)
  - results/mnist_network.png      — full network diagram
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from flashkan import KANNetwork, plot_basis, plot_activations, plot_network


def main():
    os.makedirs("results", exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    valset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    # Model
    model = KANNetwork([28 * 28, 32, 10], grid_size=8)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    # Train
    for epoch in range(10):
        model.train()
        train_correct, train_total = 0, 0
        for images, labels in trainloader:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_correct += (output.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, 28 * 28).to(device)
                labels = labels.to(device)
                output = model(images)
                val_correct += (output.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        scheduler.step()
        print(f"Epoch {epoch+1:>2}/10  "
              f"train={train_correct/train_total:.4f}  "
              f"val={val_correct/val_total:.4f}")

    print(f"\nFinal accuracy: {val_correct/val_total:.4f}")

    # Move to CPU for visualization
    model = model.cpu()

    # 1. Basis functions
    print("\nGenerating visualizations...")
    fig = plot_basis(model.layers[0])
    fig.savefig("results/mnist_basis.png", dpi=150, bbox_inches='tight')
    print("  Saved: results/mnist_basis.png")

    # 2. Learned activations — show first 6 inputs, all 10 outputs for layer 2
    fig = plot_activations(
        model.layers[1],
        in_idx=[0, 1, 2, 3, 4, 5],
        out_idx=[0, 1, 2, 3, 4],
        title="Layer 2: Learned Activations (hidden → digits)"
    )
    fig.savefig("results/mnist_activations.png", dpi=150, bbox_inches='tight')
    print("  Saved: results/mnist_activations.png")

    # 3. Network diagram
    fig = plot_network(model, title="MNIST KAN [784 → 32 → 10]")
    fig.savefig("results/mnist_network.png", dpi=150, bbox_inches='tight')
    print("  Saved: results/mnist_network.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
