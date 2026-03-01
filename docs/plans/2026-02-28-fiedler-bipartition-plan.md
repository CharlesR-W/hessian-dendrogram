# Fiedler Bipartition Dendrogram - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add recursive spectral bipartition via Fiedler vectors to build parameter-level dendrograms from the Hessian coupling graph, for the MNIST experiment.

**Architecture:** New module `src/spectral_dendrogram.py` implements the recursive bipartition algorithm.  Separate scripts handle Hessian caching (`src/recompute_hessians.py`) and the analysis pipeline (`src/run_fiedler.py`).  Visualization additions go in `src/visualize.py`.

**Tech Stack:** numpy, scipy (linalg.eigh, cluster.hierarchy), torch (checkpoint loading), matplotlib

---

### Task 1: Core spectral dendrogram module - tests

**Files:**
- Create: `tests/test_spectral_dendrogram.py`

**Step 1: Write failing tests for `fiedler_split`**

```python
# tests/test_spectral_dendrogram.py
import numpy as np
import pytest
from src.spectral_dendrogram import fiedler_split, build_fiedler_dendrogram


def test_fiedler_split_block_diagonal():
    """Block-diagonal Hessian should split cleanly along the blocks."""
    # Two 3x3 blocks with no cross-coupling
    H = np.zeros((6, 6))
    H[:3, :3] = np.array([[2.0, 0.5, 0.3],
                           [0.5, 1.5, 0.4],
                           [0.3, 0.4, 1.0]])
    H[3:, 3:] = np.array([[3.0, 0.8, 0.2],
                           [0.8, 2.0, 0.6],
                           [0.2, 0.6, 1.5]])

    left, right, lambda2 = fiedler_split(H)

    # Should split into {0,1,2} and {3,4,5} (or vice versa)
    assert set(left) | set(right) == {0, 1, 2, 3, 4, 5}
    assert len(set(left) & set(right)) == 0
    assert (set(left) == {0, 1, 2} and set(right) == {3, 4, 5}) or \
           (set(left) == {3, 4, 5} and set(right) == {0, 1, 2})
    assert lambda2 >= 0


def test_fiedler_split_weak_coupling():
    """Blocks with weak cross-coupling should still split correctly."""
    H = np.zeros((6, 6))
    H[:3, :3] = np.array([[2.0, 0.5, 0.3],
                           [0.5, 1.5, 0.4],
                           [0.3, 0.4, 1.0]])
    H[3:, 3:] = np.array([[3.0, 0.8, 0.2],
                           [0.8, 2.0, 0.6],
                           [0.2, 0.6, 1.5]])
    # Add weak cross-block coupling
    H[0, 3] = H[3, 0] = 0.01
    H[1, 4] = H[4, 1] = 0.02

    left, right, lambda2 = fiedler_split(H)

    assert (set(left) == {0, 1, 2} and set(right) == {3, 4, 5}) or \
           (set(left) == {3, 4, 5} and set(right) == {0, 1, 2})
    assert lambda2 > 0


def test_fiedler_split_two_elements():
    """Minimal split: 2 elements."""
    H = np.array([[1.0, 0.1],
                   [0.1, 2.0]])
    left, right, lambda2 = fiedler_split(H)
    assert len(left) == 1 and len(right) == 1
    assert set(left) | set(right) == {0, 1}
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/crw/Programming/Claude/hessian-dendrogram && python -m pytest tests/test_spectral_dendrogram.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.spectral_dendrogram'`

**Step 3: Commit**

```
git add tests/test_spectral_dendrogram.py
git commit -m "test: add failing tests for fiedler_split"
```

---

### Task 2: Implement `fiedler_split`

**Files:**
- Create: `src/spectral_dendrogram.py`

**Step 1: Write minimal implementation**

```python
"""Recursive spectral bipartition of the Hessian coupling graph.

Builds a dendrogram over parameters (not eigenvalues) using the Fiedler vector
of the graph Laplacian constructed from |H_ij|.  Two parameters in the same
cluster have strong second-order interaction in the loss landscape.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh


def fiedler_split(H: NDArray) -> tuple[NDArray, NDArray, float]:
    """Single spectral bipartition of parameters via the Fiedler vector.

    Constructs a coupling graph from |H_ij| (zeroed diagonal), computes the
    graph Laplacian, and splits parameters by the median of the Fiedler vector.

    Args:
        H: Symmetric (N, N) Hessian matrix.

    Returns:
        (left_indices, right_indices, lambda2) where lambda2 is the algebraic
        connectivity (2nd smallest eigenvalue of the Laplacian).
    """
    N = H.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 parameters to split")

    # Affinity: |H_ij|, zero diagonal
    A = np.abs(H)
    np.fill_diagonal(A, 0.0)

    # Graph Laplacian: L = D - A
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Fiedler vector: 2nd smallest eigenvector of L
    eigenvalues, eigenvectors = eigh(L, subset_by_index=[0, 1])
    lambda2 = eigenvalues[1]
    fiedler = eigenvectors[:, 1]

    # Median split (prevents degenerate 1-vs-(N-1) on dense graphs)
    median = np.median(fiedler)
    left = np.where(fiedler <= median)[0]
    right = np.where(fiedler > median)[0]

    # Handle exact median ties: ensure both sides non-empty
    if len(left) == 0:
        left = np.array([0])
        right = np.arange(1, N)
    elif len(right) == 0:
        right = np.array([N - 1])
        left = np.arange(0, N - 1)

    return left, right, float(lambda2)
```

**Step 2: Run tests to verify they pass**

Run: `cd /home/crw/Programming/Claude/hessian-dendrogram && python -m pytest tests/test_spectral_dendrogram.py -v`
Expected: 3 PASS

**Step 3: Commit**

```
git add src/spectral_dendrogram.py
git commit -m "feat: implement fiedler_split for spectral bipartition"
```

---

### Task 3: Tests for `build_fiedler_dendrogram`

**Files:**
- Modify: `tests/test_spectral_dendrogram.py`

**Step 1: Add failing tests for full dendrogram**

Append to `tests/test_spectral_dendrogram.py`:

```python
def test_build_fiedler_dendrogram_block_diagonal():
    """Full dendrogram from block-diagonal Hessian."""
    H = np.zeros((6, 6))
    H[:3, :3] = np.array([[2.0, 0.5, 0.3],
                           [0.5, 1.5, 0.4],
                           [0.3, 0.4, 1.0]])
    H[3:, 3:] = np.array([[3.0, 0.8, 0.2],
                           [0.8, 2.0, 0.6],
                           [0.2, 0.6, 1.5]])

    Z = build_fiedler_dendrogram(H)

    # Scipy linkage matrix: (N-1, 4) shape
    assert Z.shape == (5, 4)
    # Column 3 is cluster size; final merge should have all 6
    assert Z[-1, 3] == 6
    # Merge heights should be non-decreasing (valid linkage)
    assert np.all(np.diff(Z[:, 2]) >= -1e-12), "Merge heights not monotonic"


def test_build_fiedler_dendrogram_small():
    """Two-element Hessian produces one merge."""
    H = np.array([[1.0, 0.3],
                   [0.3, 2.0]])
    Z = build_fiedler_dendrogram(H)
    assert Z.shape == (1, 4)
    assert Z[0, 3] == 2  # merged cluster has 2 elements


def test_build_fiedler_dendrogram_identity():
    """Identity Hessian: all parameters equally coupled."""
    H = np.eye(4) + 0.1 * np.ones((4, 4))
    Z = build_fiedler_dendrogram(H)
    assert Z.shape == (3, 4)
    assert Z[-1, 3] == 4
    assert np.all(np.diff(Z[:, 2]) >= -1e-12)
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/crw/Programming/Claude/hessian-dendrogram && python -m pytest tests/test_spectral_dendrogram.py::test_build_fiedler_dendrogram_block_diagonal -v`
Expected: FAIL (function not yet implemented)

**Step 3: Commit**

```
git add tests/test_spectral_dendrogram.py
git commit -m "test: add failing tests for build_fiedler_dendrogram"
```

---

### Task 4: Implement `build_fiedler_dendrogram`

**Files:**
- Modify: `src/spectral_dendrogram.py`

**Step 1: Add tree node class and recursive implementation**

Append to `src/spectral_dendrogram.py`:

```python
from dataclasses import dataclass


@dataclass
class _TreeNode:
    """Internal node in the bipartition tree."""
    left: "_TreeNode | int"   # child node or leaf index
    right: "_TreeNode | int"
    distance: float           # 1/lambda2 (merge height)
    count: int                # total leaves in subtree


def _recursive_bipartition(
    H: NDArray,
    indices: NDArray,
    max_depth: int = 20,
    _depth: int = 0,
) -> "_TreeNode | int":
    """Recursively bipartition parameters and build a tree.

    Args:
        H: Sub-Hessian for this group of parameters.
        indices: Original parameter indices for this group.
        max_depth: Maximum recursion depth.

    Returns:
        TreeNode (internal) or int (leaf = original param index).
    """
    if len(indices) == 1:
        return int(indices[0])

    if len(indices) == 2:
        _, _, lambda2 = fiedler_split(H)
        dist = 1.0 / max(lambda2, 1e-15)
        return _TreeNode(
            left=int(indices[0]),
            right=int(indices[1]),
            distance=dist,
            count=2,
        )

    if _depth >= max_depth:
        # Force sequential merges at this distance
        node = int(indices[0])
        for i in range(1, len(indices)):
            node = _TreeNode(left=node, right=int(indices[i]),
                             distance=1e-15, count=i + 1)
        return node

    left_local, right_local, lambda2 = fiedler_split(H)
    dist = 1.0 / max(lambda2, 1e-15)

    left_indices = indices[left_local]
    right_indices = indices[right_local]
    H_left = H[np.ix_(left_local, left_local)]
    H_right = H[np.ix_(right_local, right_local)]

    left_child = _recursive_bipartition(H_left, left_indices, max_depth, _depth + 1)
    right_child = _recursive_bipartition(H_right, right_indices, max_depth, _depth + 1)

    left_count = left_child.count if isinstance(left_child, _TreeNode) else 1
    right_count = right_child.count if isinstance(right_child, _TreeNode) else 1

    return _TreeNode(
        left=left_child,
        right=right_child,
        distance=dist,
        count=left_count + right_count,
    )


def _tree_to_linkage(root: "_TreeNode | int", n: int) -> NDArray:
    """Convert bipartition tree to scipy linkage matrix Z.

    Walks the tree, assigns internal node IDs (n, n+1, ...),
    and enforces monotonicity of merge heights.

    Returns:
        Z: (n-1, 4) linkage matrix in scipy format.
    """
    if isinstance(root, int):
        return np.empty((0, 4))

    rows = []
    next_id = n  # internal node IDs start at n

    def walk(node: "_TreeNode | int") -> tuple[int, float]:
        """Returns (node_id, max_height_in_subtree)."""
        nonlocal next_id
        if isinstance(node, int):
            return node, 0.0

        left_id, left_max_h = walk(node.left)
        right_id, right_max_h = walk(node.right)

        # Enforce monotonicity: this merge must be >= all child merges
        child_max = max(left_max_h, right_max_h)
        height = max(node.distance, child_max + 1e-10)

        my_id = next_id
        next_id += 1
        rows.append([left_id, right_id, height, node.count])
        return my_id, height

    walk(root)
    return np.array(rows)


def build_fiedler_dendrogram(
    H: NDArray,
    max_depth: int = 20,
) -> NDArray:
    """Build a parameter-level dendrogram via recursive Fiedler bipartition.

    Args:
        H: Symmetric (P, P) Hessian matrix.
        max_depth: Maximum recursion depth (default 20; log2(6000) ~ 13).

    Returns:
        Scipy-format linkage matrix Z: (P-1, 4) array.
    """
    n = H.shape[0]
    if n < 2:
        return np.empty((0, 4))

    indices = np.arange(n)
    tree = _recursive_bipartition(H, indices, max_depth=max_depth)
    return _tree_to_linkage(tree, n)
```

**Step 2: Run all tests**

Run: `cd /home/crw/Programming/Claude/hessian-dendrogram && python -m pytest tests/test_spectral_dendrogram.py -v`
Expected: 6 PASS

**Step 3: Commit**

```
git add src/spectral_dendrogram.py tests/test_spectral_dendrogram.py
git commit -m "feat: implement build_fiedler_dendrogram with recursive bipartition"
```

---

### Task 5: Implement `parameter_layer_labels`

**Files:**
- Modify: `src/spectral_dendrogram.py`
- Modify: `tests/test_spectral_dendrogram.py`

**Step 1: Write failing test**

Append to `tests/test_spectral_dendrogram.py`:

```python
def test_parameter_layer_labels():
    """Labels should map each flat param index to its layer name."""
    from src.spectral_dendrogram import parameter_layer_labels
    from src.model import LeNetTiny

    model = LeNetTiny()
    labels = parameter_layer_labels(model)
    n_params = sum(p.numel() for p in model.parameters())

    assert len(labels) == n_params
    # First 200 params are conv1 weight (8*1*5*5=200)
    assert labels[0] == "features.0"
    assert labels[199] == "features.0"
    # Next 8 are conv1 bias
    assert labels[200] == "features.0"
    assert labels[207] == "features.0"
    # Conv2 starts at 208
    assert labels[208] == "features.3"
```

**Step 2: Run to verify it fails**

Run: `cd /home/crw/Programming/Claude/hessian-dendrogram && python -m pytest tests/test_spectral_dendrogram.py::test_parameter_layer_labels -v`
Expected: FAIL

**Step 3: Implement**

Append to `src/spectral_dendrogram.py`:

```python
import torch.nn as nn


def parameter_layer_labels(model: nn.Module) -> list[str]:
    """Map each flat parameter index to its layer name.

    Groups weight and bias of the same layer under the module name
    (e.g. 'features.0' for conv1 weight and bias).

    Returns:
        List of length P (total params), each entry is a layer name string.
    """
    labels = []
    for name, param in model.named_parameters():
        # Strip .weight / .bias suffix to get module name
        module_name = name.rsplit(".", 1)[0] if "." in name else name
        labels.extend([module_name] * param.numel())
    return labels
```

**Step 4: Run tests**

Run: `cd /home/crw/Programming/Claude/hessian-dendrogram && python -m pytest tests/test_spectral_dendrogram.py -v`
Expected: 7 PASS

**Step 5: Commit**

```
git add src/spectral_dendrogram.py tests/test_spectral_dendrogram.py
git commit -m "feat: add parameter_layer_labels for dendrogram coloring"
```

---

### Task 6: Hessian caching script

**Files:**
- Create: `src/recompute_hessians.py`

**Step 1: Implement the script**

```python
"""Recompute and cache full Hessian matrices from saved checkpoints.

Usage:
    cd /home/crw/Programming/Claude/hessian-dendrogram
    python -m src.recompute_hessians [--results-dir results] [--n-samples 50]
"""

import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.model import LeNetTiny
from src.train import get_hessian_subsample
from src.hessian import compute_model_hessian


def recompute_hessians(
    results_dir: str = "results",
    n_samples: int = 50,
    seed: int = 42,
):
    results_dir = Path(results_dir)
    hessians_dir = results_dir / "hessians"
    hessians_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint metadata
    with open(results_dir / "checkpoints.json") as f:
        checkpoints = json.load(f)

    # Fixed data subsample (same as original experiment)
    hessian_data, hessian_targets = get_hessian_subsample(
        n_samples=n_samples, seed=seed
    )

    model = LeNetTiny()

    for ckpt in tqdm(checkpoints, desc="Computing Hessians"):
        out_path = hessians_dir / f"step_{ckpt['step']:06d}.npy"

        if out_path.exists():
            print(f"  Skipping step {ckpt['step']} (already cached)")
            continue

        state_dict = torch.load(ckpt["state_dict_path"], weights_only=True)
        model.load_state_dict(state_dict)

        H = compute_model_hessian(model, hessian_data, hessian_targets)
        np.save(out_path, H.numpy())

        print(f"  Step {ckpt['step']:6d}: saved {out_path.name} "
              f"({H.shape[0]}x{H.shape[1]}, "
              f"{out_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache full Hessian matrices")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    recompute_hessians(args.results_dir, args.n_samples, args.seed)
```

**Step 2: Commit (no test needed - this is a pipeline script)**

```
git add src/recompute_hessians.py
git commit -m "feat: add script to recompute and cache full Hessian matrices"
```

**Step 3: Run it to populate the cache**

Run: `cd /home/crw/Programming/Claude/hessian-dendrogram && python -m src.recompute_hessians`
Expected: 25 .npy files in `results/hessians/`, ~288 MB each.
This will take several minutes.  Run in background if needed.

---

### Task 7: Fiedler pipeline script

**Files:**
- Create: `src/run_fiedler.py`

**Step 1: Implement the pipeline**

```python
"""Pipeline: load cached Hessians, compute Fiedler dendrograms, visualize.

Usage:
    cd /home/crw/Programming/Claude/hessian-dendrogram
    python -m src.run_fiedler [--results-dir results]
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from src.model import LeNetTiny
from src.spectral_dendrogram import build_fiedler_dendrogram, parameter_layer_labels
from src.visualize import (
    plot_fiedler_dendrogram_snapshots,
    plot_algebraic_connectivity,
    plot_layer_composition,
)


def run_fiedler(results_dir: str = "results"):
    results_dir = Path(results_dir)
    hessians_dir = results_dir / "hessians"
    fiedler_dir = results_dir / "fiedler"
    figures_dir = results_dir / "figures"
    fiedler_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint metadata
    with open(results_dir / "checkpoints.json") as f:
        checkpoints = json.load(f)

    # Get parameter labels
    model = LeNetTiny()
    labels = parameter_layer_labels(model)

    results = []

    for ckpt in tqdm(checkpoints, desc="Fiedler dendrograms"):
        hessian_path = hessians_dir / f"step_{ckpt['step']:06d}.npy"
        if not hessian_path.exists():
            print(f"  Skipping step {ckpt['step']} (no cached Hessian)")
            continue

        H = np.load(hessian_path)
        Z = build_fiedler_dendrogram(H)

        # Save dendrogram
        out_path = fiedler_dir / f"step_{ckpt['step']:06d}.npz"
        np.savez(out_path, linkage=Z, step=ckpt["step"], epoch=ckpt["epoch"])

        results.append({
            "step": ckpt["step"],
            "epoch": ckpt["epoch"],
            "train_loss": ckpt["train_loss"],
            "test_acc": ckpt["test_acc"],
            "linkage": Z,
        })

        print(f"  Step {ckpt['step']:6d}: dendrogram computed")

    # Visualize
    print("\nGenerating figures...")

    fig = plot_fiedler_dendrogram_snapshots(results, labels)
    fig.savefig(figures_dir / "fiedler_dendrograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fiedler_dendrograms.png")

    fig = plot_algebraic_connectivity(results)
    fig.savefig(figures_dir / "algebraic_connectivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved algebraic_connectivity.png")

    fig = plot_layer_composition(results, labels)
    fig.savefig(figures_dir / "layer_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved layer_composition.png")

    print(f"\nDone! Fiedler results in {fiedler_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fiedler bipartition analysis")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    run_fiedler(args.results_dir)
```

**Step 2: Commit**

```
git add src/run_fiedler.py
git commit -m "feat: add Fiedler bipartition pipeline script"
```

---

### Task 8: Visualization functions

**Files:**
- Modify: `src/visualize.py` (append new functions)

**Step 1: Add three new plot functions**

Append to `src/visualize.py`:

```python
from scipy.cluster.hierarchy import fcluster


def plot_fiedler_dendrogram_snapshots(
    results: list[dict],
    param_labels: list[str],
    indices: list[int] | None = None,
) -> Figure:
    """Fiedler dendrograms at selected checkpoints, leaves colored by layer.

    Args:
        results: List of dicts with 'step', 'linkage' keys.
        param_labels: List mapping flat param index -> layer name.
        indices: Which checkpoints to plot (default: ~6 evenly spaced).
    """
    if indices is None:
        n = len(results)
        indices = [0] + list(np.linspace(1, n - 1, 5, dtype=int))
        indices = sorted(set(indices))

    # Assign colors to layers
    unique_layers = list(dict.fromkeys(param_labels))  # preserve order
    layer_colors = {
        layer: plt.cm.tab10(i / max(len(unique_layers) - 1, 1))
        for i, layer in enumerate(unique_layers)
    }
    leaf_colors = [layer_colors[l] for l in param_labels]

    n_panels = len(indices)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        r = results[idx]
        Z = r["linkage"]
        if Z.shape[0] > 0:
            # Use scipy dendrogram with no labels (too many params)
            dn = dendrogram(Z, ax=ax, no_labels=True, color_threshold=0,
                           above_threshold_color="gray")
        epoch_str = f" (epoch {r['epoch']:.1f})" if 'epoch' in r else ""
        ax.set_title(f"Step {r['step']}{epoch_str}", fontsize=9)
        ax.set_ylabel("1/λ₂ (inverse connectivity)")

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=layer_colors[l], label=l) for l in unique_layers]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(unique_layers),
               fontsize=8, frameon=False)

    fig.suptitle("Fiedler Parameter Dendrogram Evolution", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    return fig


def plot_algebraic_connectivity(results: list[dict]) -> Figure:
    """Plot top-level algebraic connectivity (λ₂) over training."""
    fig, ax = plt.subplots(figsize=(10, 5))

    steps = [r["step"] for r in results]
    # λ₂ is encoded as 1/distance of the top merge
    lambda2s = [1.0 / r["linkage"][-1, 2] if r["linkage"].shape[0] > 0 else 0.0
                for r in results]

    ax.plot(steps, lambda2s, "o-", markersize=4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("λ₂ (algebraic connectivity)")
    ax.set_title("Top-level Algebraic Connectivity Over Training")
    ax.set_yscale("log")
    fig.tight_layout()
    return fig


def plot_layer_composition(
    results: list[dict],
    param_labels: list[str],
    n_clusters: int = 5,
) -> Figure:
    """Layer composition of Fiedler clusters over training.

    For each checkpoint, cuts the dendrogram into n_clusters clusters and shows
    what fraction of each cluster comes from each layer.
    """
    unique_layers = list(dict.fromkeys(param_labels))
    n_params = len(param_labels)
    label_array = np.array(param_labels)

    fig, axes = plt.subplots(1, len(results), figsize=(2.5 * len(results), 5),
                             sharey=True)
    if len(results) == 1:
        axes = [axes]

    colors = [plt.cm.tab10(i / max(len(unique_layers) - 1, 1))
              for i in range(len(unique_layers))]

    for ax, r in zip(axes, results):
        Z = r["linkage"]
        if Z.shape[0] == 0:
            continue

        cluster_ids = fcluster(Z, t=n_clusters, criterion="maxclust")

        # For each cluster, compute layer fractions
        bottoms = np.zeros(n_clusters)
        for li, layer in enumerate(unique_layers):
            heights = []
            for c in range(1, n_clusters + 1):
                mask = cluster_ids == c
                layer_mask = label_array[mask] == layer
                heights.append(layer_mask.sum() / max(mask.sum(), 1))
            ax.bar(range(1, n_clusters + 1), heights, bottom=bottoms[:n_clusters],
                   color=colors[li], label=layer if ax is axes[0] else None,
                   width=0.8)
            bottoms[:n_clusters] += heights

        ax.set_xlabel("Cluster")
        ax.set_title(f"Step {r['step']}", fontsize=8)
        ax.set_xticks(range(1, n_clusters + 1))

    axes[0].set_ylabel("Layer fraction")

    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=colors[i], label=l) for i, l in enumerate(unique_layers)]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(unique_layers),
               fontsize=8, frameon=False)
    fig.suptitle(f"Layer Composition of {n_clusters} Fiedler Clusters", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    return fig
```

**Step 2: Commit**

```
git add src/visualize.py
git commit -m "feat: add Fiedler dendrogram visualization functions"
```

---

### Task 9: Run the full pipeline

**Step 1: Ensure Hessians are cached**

Run: `cd /home/crw/Programming/Claude/hessian-dendrogram && ls results/hessians/ | wc -l`
Expected: 25 (if not, run `python -m src.recompute_hessians` first)

**Step 2: Run the Fiedler pipeline**

Run: `cd /home/crw/Programming/Claude/hessian-dendrogram && python -m src.run_fiedler`
Expected: Fiedler dendrograms computed for 25 checkpoints, 3 figures saved.

**Step 3: Verify output**

Run: `ls results/fiedler/ && ls results/figures/fiedler_* results/figures/algebraic_* results/figures/layer_*`
Expected: 25 .npz files in fiedler/, 3 new .png figures.

**Step 4: Commit results**

```
git add results/fiedler/ results/figures/fiedler_*.png results/figures/algebraic_*.png results/figures/layer_*.png
git commit -m "results: Fiedler bipartition analysis of MNIST experiment"
```

---

### Task 10: Run all tests and final verification

**Step 1: Run full test suite**

Run: `cd /home/crw/Programming/Claude/hessian-dendrogram && python -m pytest tests/ -v`
Expected: All tests pass (existing + new).

**Step 2: Check for any issues**

Run: `cd /home/crw/Programming/Claude/hessian-dendrogram && python -c "
import numpy as np
Z = np.load('results/fiedler/step_014070.npz')['linkage']
print('Final checkpoint linkage shape:', Z.shape)
print('Merge height range:', Z[:, 2].min(), '-', Z[:, 2].max())
print('Monotonic:', np.all(np.diff(Z[:, 2]) >= -1e-12))
"`
Expected: Shape (5993, 4), positive merge heights, monotonic.
