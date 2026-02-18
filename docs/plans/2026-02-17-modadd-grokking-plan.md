# Modular Addition Grokking — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train a 1-layer transformer on modular addition (a+b mod 113) and track Hessian eigenspectrum evolution through the grokking transition.

**Architecture:** Reuse existing `hessian.py` and `dendrogram.py` (model-agnostic). New source files for the transformer model, full-batch AdamW training, and the experiment pipeline. Separate marimo notebook and results directory.

**Tech Stack:** PyTorch (model/training), scipy (clustering), plotly + matplotlib (viz), marimo (notebook)

---

### Task 1: Fix hessian.py to handle integer (token) inputs

The existing `_make_loss_fn` casts `data.double()`, which breaks for integer token inputs (embeddings need LongTensor). One-line fix.

**Files:**
- Modify: `src/hessian.py:63` (the `data_f64 = data.double()` line)
- Test: `tests/test_hessian.py` (add a test with integer inputs)

**Step 1: Write failing test**

Add to `tests/test_hessian.py`:

```python
def test_hessian_with_integer_inputs():
    """Hessian computation must work when data is integer (e.g. token indices)."""
    import torch.nn as nn

    class TinyEmbedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(5, 3)
            self.linear = nn.Linear(3, 5, bias=False)

        def forward(self, x):
            return self.linear(self.embed(x))

    model = TinyEmbedModel()
    data = torch.tensor([[0, 1], [2, 3]])  # integer tokens
    targets = torch.tensor([4, 2])

    from src.hessian import compute_model_hessian
    H = compute_model_hessian(model, data, targets)

    n_params = sum(p.numel() for p in model.parameters())
    assert H.shape == (n_params, n_params)
    # Must be symmetric
    assert torch.allclose(H, H.T, atol=1e-6)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hessian.py::test_hessian_with_integer_inputs -v`
Expected: FAIL — `RuntimeError` from `data.double()` on integer tensor, or wrong results from embedding receiving float input.

**Step 3: Fix `_make_loss_fn` in `src/hessian.py`**

Change line 63 from:
```python
    data_f64 = data.double()
```
to:
```python
    data_f64 = data.double() if data.is_floating_point() else data
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hessian.py -v`
Expected: ALL PASS (including the new test and all existing tests)

**Step 5: Commit**

```bash
git add src/hessian.py tests/test_hessian.py
git commit -m "fix: handle integer (token) inputs in Hessian computation"
```

---

### Task 2: ModAddTransformer model

A 1-layer transformer for modular addition. Input: [a, b, =] (3 tokens). Predict at the = position.

**Files:**
- Create: `src/modadd_model.py`
- Create: `tests/test_modadd_model.py`

**Step 1: Write failing tests**

Create `tests/test_modadd_model.py`:

```python
import torch
from src.modadd_model import ModAddTransformer


def test_default_param_count():
    """d_model=32, p=113 should give ~11,456 parameters."""
    model = ModAddTransformer(p=113, d_model=32)
    n_params = sum(p.numel() for p in model.parameters())
    # token_embed: 114*32=3648, pos_embed: 3*32=96
    # W_Q,W_K,W_V,W_O: 4*32*32=4096, unembed: 32*113=3616
    assert n_params == 11456, f"Expected 11456 params, got {n_params}"


def test_output_shape():
    model = ModAddTransformer(p=113, d_model=32)
    tokens = torch.randint(0, 113, (8, 3))  # batch=8, seq=3
    tokens[:, 2] = 113  # = token at position 2
    out = model(tokens)
    assert out.shape == (8, 113), f"Expected (8, 113), got {out.shape}"


def test_small_model():
    """Smaller model for quick Hessian tests."""
    model = ModAddTransformer(p=7, d_model=8)
    tokens = torch.randint(0, 7, (4, 3))
    tokens[:, 2] = 7
    out = model(tokens)
    assert out.shape == (4, 7)


def test_no_batchnorm():
    """Model must not contain BatchNorm (breaks Hessian computation)."""
    model = ModAddTransformer()
    for name, module in model.named_modules():
        assert not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)), (
            f"Found BatchNorm at {name}"
        )


def test_multi_head():
    """Multi-head attention should work when d_model divisible by n_heads."""
    model = ModAddTransformer(p=113, d_model=32, n_heads=4)
    tokens = torch.randint(0, 113, (4, 3))
    tokens[:, 2] = 113
    out = model(tokens)
    assert out.shape == (4, 113)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_modadd_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.modadd_model'`

**Step 3: Implement the model**

Create `src/modadd_model.py`:

```python
"""1-layer transformer for modular addition (grokking task)."""

import torch
import torch.nn as nn


class ModAddTransformer(nn.Module):
    """Minimal transformer for a + b mod p.

    Input: [a, b, =] where a,b ∈ {0..p-1} and = is token index p.
    Output: logits over {0..p-1} from the = position.

    Architecture: token embed + pos embed → 1 self-attention layer → unembed.
    No layer norm, no MLP, no bias. Designed for full Hessian computation.
    """

    def __init__(self, p: int = 113, d_model: int = 32, n_heads: int = 1):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads

        self.token_embed = nn.Embedding(p + 1, d_model)  # p numbers + = token
        self.pos_embed = nn.Embedding(3, d_model)         # 3 positions

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.unembed = nn.Linear(d_model, p, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, S = tokens.shape
        tok_emb = self.token_embed(tokens)
        pos = torch.arange(S, device=tokens.device)
        x = tok_emb + self.pos_embed(pos)

        # Multi-head self-attention
        Q = self.W_Q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = self.W_O(torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, self.d_model))

        x = x + out  # residual
        return self.unembed(x[:, -1, :])  # read from = position
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_modadd_model.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/modadd_model.py tests/test_modadd_model.py
git commit -m "feat: add ModAddTransformer for grokking experiment"
```

---

### Task 3: Data generation and training loop

Full-batch AdamW training on all (a, b) mod p pairs. Checkpoint schedule covers memorization through grokking.

**Files:**
- Create: `src/modadd_train.py`
- Create: `tests/test_modadd_train.py`

**Step 1: Write failing tests**

Create `tests/test_modadd_train.py`:

```python
import torch
from src.modadd_train import (
    make_modadd_data,
    get_modadd_checkpoint_steps,
    train_modadd,
)
from src.modadd_model import ModAddTransformer


def test_data_generation():
    """All p² pairs generated, correct labels, deterministic split."""
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
    steps = get_modadd_checkpoint_steps(n_steps=40000)
    assert 0 in steps
    assert 40000 in steps
    assert steps == sorted(set(steps))
    # Should have 25-60 checkpoints
    assert 20 <= len(steps) <= 80, f"Got {len(steps)} checkpoints"


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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_modadd_train.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement data + training**

Create `src/modadd_train.py`:

```python
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
    """Checkpoint schedule for grokking: dense early + around transition.

    ~30 checkpoints for a 40K step run.
    """
    steps = {0, 50, 100, 500}
    steps.update(range(1000, min(10001, n_steps + 1), 1000))
    steps.update(range(10000, n_steps + 1, 2000))
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
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_modadd_train.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/modadd_train.py tests/test_modadd_train.py
git commit -m "feat: add modular addition data generation and training loop"
```

---

### Task 4: End-to-end pipeline (run_modadd.py)

Orchestrates: train → Hessian spectra → static visualizations. Mirrors `run_experiment.py` structure.

**Files:**
- Create: `src/run_modadd.py`

**Step 1: Implement pipeline**

Create `src/run_modadd.py`:

```python
"""End-to-end pipeline for modular addition grokking + Hessian analysis."""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.modadd_model import ModAddTransformer
from src.modadd_train import make_modadd_data, train_modadd
from src.hessian import compute_model_hessian, compute_eigenspectrum
from src.visualize import (
    plot_spectrum_evolution,
    plot_dendrogram_snapshots,
    plot_gap_barcode,
    plot_cluster_heatmap,
    plot_summary_stats,
)


def run(
    p: int = 113,
    d_model: int = 32,
    n_heads: int = 1,
    n_steps: int = 40000,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    train_fraction: float = 0.3,
    hessian_n_samples: int = 200,
    n_save_vectors: int = 50,
    seed: int = 42,
    results_dir: str = "results_modadd",
):
    results_dir = Path(results_dir)
    checkpoint_dir = results_dir / "checkpoints"
    spectra_dir = results_dir / "spectra"
    figures_dir = results_dir / "figures"
    for d in [checkpoint_dir, spectra_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    # ---- Phase 1: Train ----
    print("=" * 60)
    print("Phase 1: Training")
    print("=" * 60)

    model = ModAddTransformer(p=p, d_model=d_model, n_heads=n_heads)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"Model: ModAddTransformer(p={p}, d_model={d_model}), {n_params} parameters")

    checkpoint_results = train_modadd(
        model=model,
        p=p,
        n_steps=n_steps,
        lr=lr,
        weight_decay=weight_decay,
        train_fraction=train_fraction,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )

    with open(results_dir / "checkpoints.json", "w") as f:
        json.dump(checkpoint_results, f, indent=2)
    print(f"Saved {len(checkpoint_results)} checkpoints")

    # ---- Phase 2: Compute Hessian spectra ----
    print("\n" + "=" * 60)
    print("Phase 2: Hessian Eigenspectra")
    print("=" * 60)

    # Fixed Hessian subsample from training data
    train_tok, train_lab, _, _ = make_modadd_data(
        p=p, train_fraction=train_fraction, seed=seed,
    )
    rng = torch.Generator().manual_seed(seed + 1)
    n_sub = min(hessian_n_samples, len(train_tok))
    sub_idx = torch.randperm(len(train_tok), generator=rng)[:n_sub]
    hessian_data = train_tok[sub_idx]
    hessian_targets = train_lab[sub_idx]
    print(f"Hessian subsample: {n_sub} training pairs")

    spectra = []
    for ckpt in tqdm(checkpoint_results, desc="Computing Hessian spectra"):
        state_dict = torch.load(ckpt["state_dict_path"], weights_only=True)
        model.load_state_dict(state_dict)

        H = compute_model_hessian(model, hessian_data, hessian_targets)
        eigenvalues, eigenvectors = compute_eigenspectrum(H, n_save_vectors=n_save_vectors)

        spec_path = spectra_dir / f"step_{ckpt['step']:06d}.npz"
        np.savez(
            spec_path,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            step=ckpt["step"],
        )

        spectra.append({
            "step": ckpt["step"],
            "train_loss": ckpt["train_loss"],
            "test_acc": ckpt["test_acc"],
            "eigenvalues": eigenvalues,
        })

        n_neg = np.sum(eigenvalues < 0)
        print(f"  Step {ckpt['step']:6d}: {n_neg} negative eigenvalues, "
              f"range [{eigenvalues.min():.2f}, {eigenvalues.max():.2f}]")

    # ---- Phase 3: Visualize ----
    print("\n" + "=" * 60)
    print("Phase 3: Visualization")
    print("=" * 60)

    import matplotlib
    matplotlib.use("Agg")

    fig = plot_spectrum_evolution(spectra)
    fig.savefig(figures_dir / "spectrum_evolution.png", dpi=150, bbox_inches="tight")
    print("  Saved spectrum_evolution.png")

    fig = plot_dendrogram_snapshots(spectra)
    fig.savefig(figures_dir / "dendrogram_snapshots.png", dpi=150, bbox_inches="tight")
    print("  Saved dendrogram_snapshots.png")

    fig = plot_gap_barcode(spectra, k=15)
    fig.savefig(figures_dir / "gap_barcode.png", dpi=150, bbox_inches="tight")
    print("  Saved gap_barcode.png")

    fig = plot_cluster_heatmap(spectra)
    fig.savefig(figures_dir / "cluster_heatmap.png", dpi=150, bbox_inches="tight")
    print("  Saved cluster_heatmap.png")

    fig = plot_summary_stats(spectra)
    fig.savefig(figures_dir / "summary_stats.png", dpi=150, bbox_inches="tight")
    print("  Saved summary_stats.png")

    import matplotlib.pyplot as plt
    plt.close("all")

    print("\n" + "=" * 60)
    print(f"Done! All results in {results_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    run()
```

**Step 2: Verify imports resolve**

Run: `uv run python -c "from src.run_modadd import run; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/run_modadd.py
git commit -m "feat: add end-to-end pipeline for modular addition experiment"
```

---

### Task 5: Marimo notebook (modadd_app.py)

Interactive visualization of grokking + Hessian results. Same structure as `app.py` but reads from `results_modadd/` and adds grokking-specific views (train vs test accuracy showing the grokking transition).

**Files:**
- Create: `modadd_app.py`

**Step 1: Create the notebook**

Create `modadd_app.py` — follow exact same cell structure as `app.py` but:

1. Load from `results_modadd/` instead of `results/`
2. Header: "Modular Addition Grokking" not "LeNet-tiny on MNIST"
3. Add **grokking curve** cell (train loss + test loss + train acc + test acc on same plot, dual y-axes)
4. Checkpoints have `train_acc` field (not just `test_acc`)
5. Parameter count differs (~11,456 not 5,994) — update colorbar ticktext
6. All other cells (spectrum evolution, dendrogram, GIF, heatmap, n(ε), gaps, summary, histogram) carry over with minor label changes

The notebook must:
- Use `# /// script` metadata with all dependencies
- Use marimo sandbox mode (`--sandbox`)
- Use `plotly_dark` template + dark matplotlib styling
- Use the same PALETTE
- NOT import from `src/` (inline any needed logic, since sandbox can't access local modules)

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy>=1.24",
#     "plotly>=5.18",
#     "scipy>=1.10",
#     "matplotlib>=3.7",
#     "pillow>=10.0",
# ]
#
# [tool.marimo]
# width = "medium"
# theme = "dark"
# ///
```

Key differences from app.py:
- Data loading cell reads `results_modadd/` and includes `train_acc` from checkpoints
- First visualization cell: "Grokking Curve" — plotly figure with train_loss, test_loss (log scale left y-axis) + train_acc, test_acc (right y-axis) on dual axes
- Summary stats panel: 6 subplots (Train Loss, Test Loss, Train Accuracy, Test Accuracy, Hessian Trace, Spectral Entropy) — replacing some MNIST-specific ones
- Colorbar uses model's actual param count for max tick
- Footer text describes modular addition experiment

**Step 2: Verify export**

Run: `marimo export html modadd_app.py --sandbox --output /dev/null 2>&1 | head -5`
Expected: Clean output, no errors

**Step 3: Commit**

```bash
git add modadd_app.py
git commit -m "feat: add marimo notebook for modular addition grokking visualization"
```

---

### Task 6: Update .gitignore and run experiment

**Files:**
- Modify: `.gitignore` (add `results_modadd/`)

**Step 1: Update .gitignore**

Add `results_modadd/` to `.gitignore` (same as `results/`).

**Step 2: Run the experiment**

```bash
nohup uv run python -c "from src.run_modadd import run; run()" > /tmp/modadd-experiment.log 2>&1 &
```

Monitor with: `tail -f /tmp/modadd-experiment.log`

Expected runtime: ~5-9 hours (training ~30 min + ~30 Hessian computations at ~10 min each).

**Step 3: Commit .gitignore**

```bash
git add .gitignore
git commit -m "chore: add results_modadd/ to .gitignore"
```

---

## Notes

- The experiment is long-running (~5-9 hours). Training is fast (full batch on 3,831 pairs). The bottleneck is Hessian computation: ~11K × 11K matrix per checkpoint.
- To scale up: change `d_model=48` or `d_model=128` in the `run()` call. At d_model=128, the Hessian computation becomes infeasible for full eigendecomposition — would need Lanczos (top-k only). That's a separate follow-up.
- The grokking transition timing varies with hyperparameters. If grokking doesn't occur by step 40K, increase `n_steps` or adjust `weight_decay`.
- Monitor training progress: look for train_acc reaching ~1.0 while test_acc remains at ~1/p ≈ 0.009, then a sudden jump in test_acc — that's the grokking transition.
