# Modular Addition Grokking — Hessian Eigenspectrum Analysis

**Date:** 2026-02-17

## Goal

Replicate the Hessian eigenspectrum dendrogram analysis (already done for LeNet-tiny/MNIST) on the standard modular addition grokking task.  The grokking transition — where the model suddenly generalizes after long memorization — should produce dramatic spectral restructuring visible in the dendrograms and cluster counts.

## Model

1-layer transformer, single attention head, for `a + b mod p`:

- Token embedding: `p × d_model` (p=113 prime)
- Positional embedding: `3 × d_model` (positions: a, b, =)
- Single self-attention: W_Q, W_K, W_V, W_O each `d_model × d_model`
- Unembedding: `d_model × p`
- **Default d_model=32** (~11,400 params), parameterized for scaling to 48/128

No layer norm (following simpler grokking setups).  ReLU activation after attention if needed for grokking to occur at small d_model — test empirically.

## Data

- All p² = 12,769 pairs `(a, b)` with label `(a+b) mod p`
- Input format: 3-token sequence [a, b, =], predict next token
- Train/test split: 30% train (~3,831 pairs), 70% test
- Deterministic split (seeded)

## Training

- **Optimizer:** AdamW, lr=1e-3, weight_decay=1.0
- **Full-batch** training (all training pairs per step, 1 step = 1 epoch)
- **Duration:** 40,000 steps
- **Loss:** Cross-entropy on the output token at position 2 (the = position)

### Checkpoint Schedule

Dense around expected grokking transition, sparse elsewhere:
- Steps 0, 10, 50, 100, 200, 500
- Every 100 steps from 1K to 5K
- Every 500 steps from 5K to 15K
- Every 1000 steps from 15K to 40K
- ~75-80 checkpoints total (more than MNIST because training is longer and transition timing is uncertain)

## Hessian Computation

- Reuse existing `src/hessian.py` (batched vmap HVP)
- Fixed subsample: 200 training pairs (deterministic)
- Full eigendecomposition of ~11K × 11K Hessian
- Save top/bottom 50 eigenvectors + all eigenvalues per checkpoint
- ~130M entries per Hessian at d_model=32, ~1 GB float64

## Project Structure

```
hessian-dendrogram/
├── src/
│   ├── modadd_model.py      # 1-layer transformer
│   ├── modadd_train.py      # Data gen + full-batch AdamW training
│   └── run_modadd.py        # End-to-end pipeline
├── modadd_app.py             # Marimo notebook
├── results_modadd/
│   ├── checkpoints/
│   ├── spectra/
│   └── figures/
└── tests/
    ├── test_modadd_model.py
    └── test_modadd_train.py
```

Reuses: `src/hessian.py`, `src/dendrogram.py`, `src/visualize.py` (partially).

## Notebook Visualizations (modadd_app.py)

1. **Grokking curve** — train + test loss/accuracy, with grokking transition highlighted
2. **Spectrum evolution** — eigenvalue trajectories with grokking transition marked
3. **Truncated dendrogram + slider** — top-30 merges, dark theme
4. **Dendrogram GIF** — animated evolution over all checkpoints
5. **Cluster heatmap** — n(ε) over training, log color scale
6. **n(ε) curves** — before / during / after grokking
7. **Spectral gaps barcode** — top-10 gaps over training
8. **Summary stats** — loss, accuracy, trace, entropy, max eigenvalue, #negative evals

## Configurable Parameters

| Parameter | Default | Scale-up |
|-----------|---------|----------|
| p | 113 | — |
| d_model | 32 | 48, 128 |
| n_heads | 1 | 1-4 |
| train_fraction | 0.3 | — |
| lr | 1e-3 | — |
| weight_decay | 1.0 | — |
| n_steps | 40000 | — |
| hessian_n_samples | 200 | — |
