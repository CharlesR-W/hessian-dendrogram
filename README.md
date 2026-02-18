# Hessian Dendrograms

**Tracking circuit formation in neural networks through Hessian eigenspectrum analysis.**

The loss landscape Hessian encodes how a network's learned representations are structured.  By computing the full eigenspectrum at checkpoints throughout training, we can build *spectral dendrograms* — hierarchical clusterings of eigenvalues that reveal when and how distinct computational circuits emerge.

## Key idea

Coarse-grain the Hessian eigenvalues at resolution $\varepsilon$: eigenvalues differing by less than $\varepsilon$ are grouped into the same cluster.  As $\varepsilon$ increases, clusters merge — producing a dendrogram whose branch points correspond to spectral gaps separating distinct curvature scales in the loss landscape.

**Physical interpretation:** A tight cluster of eigenvalues = a circuit (perturbing along any direction within the cluster costs similar loss).  A spectral gap = a boundary between circuits.  Watching dendrograms evolve over training reveals circuit formation dynamics.

## Experiments

### 1. LeNet-tiny on MNIST (complete)

A minimal conv-net (~6K parameters) trained on MNIST with SGD.  Full Hessian eigendecomposition at 25 checkpoints.

- **Model:** `Conv2d(1,8,5) → ReLU → MaxPool → Conv2d(8,16,5) → ReLU → MaxPool → Linear(256,10)`
- **Training:** SGD with momentum, 30 epochs, ~25 checkpoints
- **Hessian:** Full 6K×6K matrix, all eigenvalues + top/bottom 50 eigenvectors

Results in `results/figures/` — spectrum evolution, dendrograms, cluster heatmaps, spectral gap barcodes, and learned filter visualizations.

### 2. Modular addition grokking (in progress)

A 1-layer transformer learning `a + b mod 113`, the canonical *grokking* task where models suddenly generalize long after memorizing the training set (Power et al. 2022, Nanda et al. 2023).

- **Model:** 1-layer transformer, d_model=128, 4 attention heads (~95K parameters)
- **Training:** Full-batch AdamW (lr=1e-3, weight_decay=1.0), 50K steps, 30% train split
- **Hessian:** Lanczos iteration for top/bottom 200 eigenvalues via Hessian-vector products (full matrix would be 72GB)

The grokking transition — memorization followed by sudden generalization — should produce dramatic spectral restructuring: the dendrogram should reorganize as the model transitions from a memorization solution to Fourier circuits computing modular arithmetic.

## Visualizations

Interactive marimo notebooks for both experiments:

```bash
# MNIST experiment
marimo run app.py --sandbox

# Modular addition grokking
marimo run modadd_app.py --sandbox
```

Each notebook includes:
- Eigenvalue spectrum evolution over training
- Animated dendrogram GIFs
- Cluster count n(ε) heatmaps (resolution × training step)
- Top spectral gap barcodes
- Summary statistics (loss, accuracy, trace, entropy, max eigenvalue)

## Project structure

```
src/
├── hessian.py          # Full Hessian + Lanczos eigenvalue computation
├── dendrogram.py       # Single-linkage clustering on eigenvalues
├── visualize.py        # Matplotlib visualization functions
├── model.py            # LeNet-tiny (MNIST)
├── train.py            # MNIST training loop
├── run_experiment.py   # MNIST end-to-end pipeline
├── modadd_model.py     # 1-layer transformer (grokking)
├── modadd_train.py     # Modular addition data + training
└── run_modadd.py       # Grokking end-to-end pipeline
```

## Running experiments

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Run MNIST experiment (~2 hours)
uv run python -m src.run_experiment

# Run modular addition experiment (~9 hours: training + Lanczos)
uv run python -m src.run_modadd
```

## References

- Power et al. (2022). *Grokking: Generalization beyond overfitting on small algorithmic datasets.*
- Nanda et al. (2023). *Progress measures for grokking via mechanistic interpretability.*
- Sagun et al. (2017). *Eigenvalues of the Hessian in deep learning.*
- Ghorbani et al. (2019). *An investigation into neural net optimization via Hessian eigenvalue density.*

## License

MIT

---

*Written with Claude*
