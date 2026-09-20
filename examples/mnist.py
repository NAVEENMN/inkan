"""Train a KAN classifier on MNIST.

Usage:
    pip install inkan torchvision tqdm
    python examples/mnist.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from inkan import KANNetwork


def main() -> None:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    valset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    model = KANNetwork([28 * 28, 64, 10])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        with tqdm(trainloader, desc=f"Epoch {epoch+1}") as pbar:
            for images, labels in pbar:
                images = images.view(-1, 28 * 28).to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                accuracy = (output.argmax(dim=1) == labels).float().mean()
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{accuracy.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                )

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, 28 * 28).to(device)
                labels = labels.to(device)
                output = model(images)
                val_loss += criterion(output, labels).item() * labels.size(0)
                val_correct += (output.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_accuracy = val_correct / val_total
        scheduler.step()

        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    main()
