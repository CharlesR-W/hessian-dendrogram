# Hessian Dendrograms

**Companion code for [A Diachronic View of Circuits](https://crw.dev/posts/A-Diachronic-View-of-Circuits/).**

Tracks circuit formation in neural networks through Hessian eigenspectrum analysis during training.  The core idea: coarse-grain the Hessian eigenvalues at resolution $\varepsilon$, build a dendrogram from the resulting clusters, and watch spectral gaps appear as circuits emerge.

> **Research companion code.**  This repository implements the numerical experiments and visualizations for the blog post series on spectral approaches to mechanistic interpretability.  It is a work in progress - the code is functional and produces the reported results, but should be understood as research infrastructure rather than a polished library.  Developed during [MATS](https://www.matsprogram.org/) 9.0.

## Experiments

### 1. LeNet-tiny on MNIST (complete)

A minimal conv-net (~6K parameters) trained on MNIST with SGD.  Full Hessian eigendecomposition at 25 checkpoints throughout training.

- **Model:** `Conv2d(1,8,5) → ReLU → MaxPool → Conv2d(8,16,5) → ReLU → MaxPool → Linear(256,10)`
- **Training:** SGD with momentum, 30 epochs
- **Hessian:** Full 6K×6K matrix, all eigenvalues + top/bottom 50 eigenvectors
- **Results:** `results/figures/` - spectrum evolution, dendrograms, cluster heatmaps, spectral gap barcodes, learned filter visualizations

### 2. Modular addition grokking (complete)

A 1-layer transformer learning `a + b mod 113`, the canonical grokking task where models suddenly generalize long after memorizing (Power et al. 2022, Nanda et al. 2023).

- **Model:** 1-layer transformer, d_model=128, 4 attention heads (~95K parameters)
- **Training:** Full-batch AdamW, 150K steps, 30% train split
- **Hessian:** Lanczos iteration for top/bottom 200 eigenvalues via Hessian-vector products (full matrix would be 72GB)
- **Results:** `results_modadd/figures/` - spectral restructuring during the grokking transition

### 3. Fiedler bipartition dendrograms (complete)

A complementary approach: instead of clustering eigenvalues, recursively bipartition *parameters* using the Fiedler vector of the Hessian coupling graph $|H_{ij}|$.  Two parameters in the same cluster have strong second-order interactions.

- **Results:** `results/figures/` - Fiedler dendrograms, algebraic connectivity over training, layer composition heatmaps

## Interactive visualization (marimo notebooks)

The experiment pipelines produce raw data; the marimo notebooks provide interactive exploration.  These are **visualization tools**, not experiments - they load pre-computed results from `results/` and `results_modadd/`.

```bash
marimo run app.py --sandbox          # MNIST eigenspectra + dendrograms
marimo run modadd_app.py --sandbox   # Grokking eigenspectra + dendrograms
marimo run fiedler_app.py --sandbox  # Fiedler bipartition analysis
```

## Running experiments

```bash
uv sync

# Unit tests
uv run pytest tests/ -v

# MNIST experiment (~2 hours: train + full Hessian at each checkpoint)
uv run python -m src.run_experiment

# Grokking experiment (~9 hours: training + Lanczos at each checkpoint)
uv run python -m src.run_modadd

# Fiedler dendrograms (requires cached Hessians from MNIST experiment)
uv run python -m src.run_fiedler
```

## Project structure

```
src/
├── hessian.py              # Full Hessian + Lanczos eigenvalue computation
├── dendrogram.py           # Single-linkage clustering on eigenvalues
├── spectral_dendrogram.py  # Fiedler bipartition + recursive parameter clustering
├── visualize.py            # Matplotlib visualization functions
├── model.py                # LeNet-tiny (MNIST)
├── train.py                # MNIST training loop
├── run_experiment.py       # MNIST end-to-end pipeline
├── modadd_model.py         # 1-layer transformer (grokking)
├── modadd_train.py         # Modular addition data + training
├── run_modadd.py           # Grokking end-to-end pipeline
└── run_fiedler.py          # Fiedler bipartition pipeline
```

## Related posts

- [A Diachronic View of Circuits](https://crw.dev/posts/A-Diachronic-View-of-Circuits/) - theoretical framework for tracking circuits via Hessian spectra
- [Toy Models of Neural Darwinism](https://crw.dev/posts/Toy-Models-of-Neural-Darwinism/) - competition dynamics between circuits
- [The Spectral Structure of Natural Data](https://crw.dev/posts/The-Spectral-Structure-of-Natural-Data/) - data-side spectral geometry

## References

- Power et al. (2022). *Grokking: Generalization beyond overfitting on small algorithmic datasets.*
- Nanda et al. (2023). *Progress measures for grokking via mechanistic interpretability.*
- Sagun et al. (2017). *Eigenvalues of the Hessian in deep learning.*
- Ghorbani et al. (2019). *An investigation into neural net optimization via Hessian eigenvalue density.*

## License

MIT

---

*Developed during MATS 9.0.  Written with Claude.*
