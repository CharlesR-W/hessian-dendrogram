# tests/test_train.py
import torch
from pathlib import Path
from src.train import (
    get_checkpoint_steps,
    get_mnist_loaders,
    train_with_checkpoints,
)
from src.model import LeNetTiny


def test_checkpoint_steps():
    """Verify checkpoint schedule matches design."""
    steps = get_checkpoint_steps(n_epochs=30, steps_per_epoch=469)
    # Must include step 0 (init)
    assert 0 in steps
    # Must include sub-epoch steps
    assert 10 in steps
    assert 50 in steps
    assert 100 in steps
    # Must include end-of-epoch steps
    assert 469 in steps  # end of epoch 1
    # Sorted, no duplicates
    assert steps == sorted(set(steps))


def test_train_short(tmp_path):
    """Quick smoke test: train for 2 epochs, verify checkpoints saved."""
    model = LeNetTiny()
    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    results = train_with_checkpoints(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=2,
        lr=0.01,
        momentum=0.9,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    # Must have at least the init checkpoint + epoch-end checkpoints
    assert len(results) >= 3
    # Each result has required fields
    for r in results:
        assert "step" in r
        assert "epoch" in r
        assert "train_loss" in r
        assert "test_acc" in r
        assert "state_dict_path" in r
        assert Path(r["state_dict_path"]).exists()
