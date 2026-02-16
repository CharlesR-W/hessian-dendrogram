"""Training loop with configurable checkpoint schedule."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm


def get_mnist_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Load MNIST train and test sets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    return train_loader, test_loader


def get_hessian_subsample(
    data_dir: str = "./data",
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a fixed subsample of training data for Hessian computation."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)

    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_set), generator=rng)[:n_samples]
    subset = Subset(train_set, indices.tolist())
    loader = DataLoader(subset, batch_size=n_samples, shuffle=False)
    data, targets = next(iter(loader))
    return data, targets


def get_checkpoint_steps(n_epochs: int, steps_per_epoch: int) -> list[int]:
    """Compute the global step numbers at which to save checkpoints.

    Schedule: step 0 (init), sub-epoch [10, 50, 100, 200, 500],
    end of epochs 1-10 (every epoch), epochs 12-30 (every 2 epochs).
    """
    steps = {0}

    # Sub-epoch checkpoints (within first epoch)
    for s in [10, 50, 100, 200, 500]:
        if s < steps_per_epoch:
            steps.add(s)

    # End-of-epoch checkpoints
    for epoch in range(1, min(n_epochs + 1, 11)):
        steps.add(epoch * steps_per_epoch)
    for epoch in range(12, n_epochs + 1, 2):
        steps.add(epoch * steps_per_epoch)

    return sorted(steps)


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader) -> float:
    """Compute test accuracy."""
    model.eval()
    correct = 0
    total = 0
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return correct / total


def train_with_checkpoints(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int = 30,
    lr: float = 0.01,
    momentum: float = 0.9,
    checkpoint_dir: str | Path = "results/checkpoints",
) -> list[dict]:
    """Train model and save checkpoints at scheduled steps.

    Returns list of dicts with checkpoint metadata.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    checkpoint_steps = set(get_checkpoint_steps(n_epochs, steps_per_epoch))
    results = []

    global_step = 0
    running_loss = 0.0
    loss_count = 0

    def save_checkpoint(step, epoch, train_loss):
        test_acc = evaluate(model, test_loader)
        path = checkpoint_dir / f"step_{step:06d}.pt"
        torch.save(model.state_dict(), path)
        result = {
            "step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            "test_acc": test_acc,
            "state_dict_path": str(path),
        }
        results.append(result)
        return result

    # Checkpoint at init (step 0)
    save_checkpoint(0, 0, float("nan"))

    for epoch in range(1, n_epochs + 1):
        model.train()
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_count += 1
            global_step += 1

            if global_step in checkpoint_steps:
                avg_loss = running_loss / loss_count if loss_count > 0 else float("nan")
                r = save_checkpoint(global_step, epoch, avg_loss)
                tqdm.write(
                    f"  Checkpoint step={global_step} epoch={epoch} "
                    f"loss={avg_loss:.4f} acc={r['test_acc']:.4f}"
                )
                running_loss = 0.0
                loss_count = 0

    return results
