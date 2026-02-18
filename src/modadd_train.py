"""Data generation and full-batch training for modular addition."""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm


def make_modadd_data(
    p: int = 113,
    train_fraction: float = 0.3,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate all (a + b) mod p data with deterministic train/test split.

    Returns (train_tokens, train_labels, test_tokens, test_labels).
    Tokens are shape (N, 3): [a, b, =_token] where =_token = p.
    Labels are shape (N,): (a + b) mod p.
    """
    all_tokens = []
    all_labels = []
    for a in range(p):
        for b in range(p):
            all_tokens.append([a, b, p])  # p is the = token
            all_labels.append((a + b) % p)

    all_tokens = torch.tensor(all_tokens, dtype=torch.long)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(all_tokens), generator=rng)
    n_train = int(len(all_tokens) * train_fraction)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    return (
        all_tokens[train_idx], all_labels[train_idx],
        all_tokens[test_idx], all_labels[test_idx],
    )


def get_modadd_checkpoint_steps(n_steps: int) -> list[int]:
    """Checkpoint schedule for grokking: dense early, moderate mid, sparse late.

    ~35 checkpoints for a 150K step run.
    """
    steps = {0, 50, 100, 500}
    steps.update(range(1000, min(5001, n_steps + 1), 1000))
    steps.update(range(5000, min(20001, n_steps + 1), 2500))
    steps.update(range(20000, min(80001, n_steps + 1), 5000))
    steps.update(range(80000, n_steps + 1, 10000))
    steps.add(n_steps)
    return sorted(s for s in steps if s <= n_steps)


@torch.no_grad()
def _accuracy(model: nn.Module, tokens: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    logits = model(tokens)
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def train_modadd(
    model: nn.Module,
    p: int = 113,
    n_steps: int = 40000,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    train_fraction: float = 0.3,
    checkpoint_dir: str | Path = "results_modadd/checkpoints",
    seed: int = 42,
) -> list[dict]:
    """Full-batch AdamW training with checkpointing.

    Returns list of checkpoint metadata dicts.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_tok, train_lab, test_tok, test_lab = make_modadd_data(
        p=p, train_fraction=train_fraction, seed=seed,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    checkpoint_steps = set(get_modadd_checkpoint_steps(n_steps))
    results = []

    def save_checkpoint(step):
        train_loss = criterion(model(train_tok), train_lab).item()
        train_acc = _accuracy(model, train_tok, train_lab)
        test_acc = _accuracy(model, test_tok, test_lab)
        path = checkpoint_dir / f"step_{step:06d}.pt"
        torch.save(model.state_dict(), path)
        result = {
            "step": step,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "state_dict_path": str(path),
        }
        results.append(result)
        return result

    # Step 0 checkpoint
    model.eval()
    save_checkpoint(0)

    for step in tqdm(range(1, n_steps + 1), desc="Training"):
        model.train()
        optimizer.zero_grad()
        logits = model(train_tok)
        loss = criterion(logits, train_lab)
        loss.backward()
        optimizer.step()

        if step in checkpoint_steps:
            r = save_checkpoint(step)
            tqdm.write(
                f"  Step {step:6d}: loss={r['train_loss']:.4f} "
                f"train_acc={r['train_acc']:.3f} test_acc={r['test_acc']:.3f}"
            )

    return results
