import torch
from src.modadd_train import (
    make_modadd_data,
    get_modadd_checkpoint_steps,
    train_modadd,
)
from src.modadd_model import ModAddTransformer


def test_data_generation():
    """All p^2 pairs generated, correct labels, deterministic split."""
    train_tok, train_lab, test_tok, test_lab = make_modadd_data(p=7, train_fraction=0.3)
    total = len(train_tok) + len(test_tok)
    assert total == 49, f"Expected 49 pairs for p=7, got {total}"
    # Labels are correct
    for tok, lab in zip(train_tok, train_lab):
        assert lab == (tok[0] + tok[1]) % 7
    # = token is at position 2
    assert (train_tok[:, 2] == 7).all()
    assert (test_tok[:, 2] == 7).all()


def test_data_deterministic():
    """Same seed gives same split."""
    d1 = make_modadd_data(p=7, seed=42)
    d2 = make_modadd_data(p=7, seed=42)
    assert torch.equal(d1[0], d2[0])


def test_checkpoint_steps():
    steps = get_modadd_checkpoint_steps(n_steps=150000)
    assert 0 in steps
    assert 150000 in steps
    assert steps == sorted(set(steps))
    # Should have 30-50 checkpoints for 150K steps
    assert 25 <= len(steps) <= 50, f"Got {len(steps)} checkpoints"


def test_train_smoke(tmp_path):
    """Quick smoke test: train tiny model for 20 steps."""
    model = ModAddTransformer(p=7, d_model=8)
    results = train_modadd(
        model=model,
        p=7,
        n_steps=20,
        lr=1e-3,
        weight_decay=1.0,
        checkpoint_dir=tmp_path / "ckpt",
        seed=42,
    )
    assert len(results) >= 2  # at least init + final
    for r in results:
        assert "step" in r
        assert "train_loss" in r
        assert "train_acc" in r
        assert "test_acc" in r
        assert "state_dict_path" in r
