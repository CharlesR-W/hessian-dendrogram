"""Visualization functions for Hessian spectral dendrogram analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram, fcluster
from src.dendrogram import build_dendrogram, cluster_count_curve, extract_top_gaps


def plot_spectrum_evolution(spectra: list[dict]) -> Figure:
    """Plot eigenvalue spectra across training checkpoints.

    Single panel showing eigenvalues on log scale.  Works for both full
    spectra and Lanczos top-k (where negative eigenvalues are absent).
    """
    has_neg = any(np.any(s["eigenvalues"] < 0) for s in spectra)

    if has_neg:
        fig, (ax_pos, ax_neg) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax_pos = plt.subplots(1, 1, figsize=(10, 5))

    cmap = plt.cm.viridis
    n = len(spectra)

    for i, spec in enumerate(spectra):
        evals = spec["eigenvalues"]
        color = cmap(i / max(n - 1, 1))
        label = f"step {spec['step']}"

        pos = evals[evals > 0]
        if len(pos) > 0:
            ax_pos.semilogy(range(len(pos)), pos, color=color, alpha=0.7,
                           linewidth=0.8, label=label)

        if has_neg:
            neg = evals[evals < 0]
            if len(neg) > 0:
                ax_neg.semilogy(range(len(neg)), np.abs(neg[::-1]), color=color,
                               alpha=0.7, linewidth=0.8, label=label)

    ax_pos.set_title("Positive eigenvalues (top-k)" if not has_neg else "Positive eigenvalues")
    ax_pos.set_xlabel("Index")
    ax_pos.set_ylabel("Eigenvalue")
    ax_pos.legend(fontsize=6, ncol=2)

    if has_neg:
        ax_neg.set_title("Negative eigenvalues (|λ|)")
        ax_neg.set_xlabel("Index (largest |λ| first)")
        ax_neg.set_ylabel("|Eigenvalue|")

    fig.tight_layout()
    return fig


def plot_dendrogram_snapshots(
    spectra: list[dict],
    indices: list[int] | None = None,
) -> Figure:
    """Plot dendrograms at selected checkpoints."""
    if indices is None:
        # Pick ~6 evenly spaced
        n = len(spectra)
        indices = [0] + list(np.linspace(1, n - 1, 5, dtype=int))
        indices = sorted(set(indices))

    n_panels = len(indices)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        spec = spectra[idx]
        Z = build_dendrogram(spec["eigenvalues"])
        if Z.shape[0] > 0:
            dendrogram(Z, ax=ax, no_labels=True, color_threshold=0)
        epoch_str = f" (epoch {spec['epoch']:.1f})" if 'epoch' in spec else ""
        ax.set_title(f"Step {spec['step']}{epoch_str}", fontsize=9)
        ax.set_ylabel("Gap size (ε)")

    fig.suptitle("Dendrogram Evolution", fontsize=12)
    fig.tight_layout()
    return fig


def plot_gap_barcode(spectra: list[dict], k: int = 20) -> Figure:
    """Track the top-k largest spectral gaps over training."""
    fig, ax = plt.subplots(figsize=(12, 6))

    steps = [s["step"] for s in spectra]
    gap_matrix = np.array([extract_top_gaps(s["eigenvalues"], k=k) for s in spectra])

    for i in range(min(k, gap_matrix.shape[1])):
        ax.semilogy(steps, gap_matrix[:, i], marker=".", markersize=3,
                    linewidth=0.8, alpha=0.7, label=f"Gap {i+1}")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Gap size")
    ax.set_title(f"Top-{k} Spectral Gaps Over Training")
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    return fig


def plot_cluster_heatmap(spectra: list[dict], n_eps: int = 100) -> Figure:
    """Heatmap of cluster count n(ε) over training and resolution."""
    # Compute global epsilon range
    all_evals = np.concatenate([s["eigenvalues"] for s in spectra])
    gaps = np.diff(np.sort(all_evals))
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No gaps found", ha="center", va="center")
        return fig

    eps_lo = max(gaps.min() * 0.5, 1e-12)
    eps_hi = gaps.max() * 2.0
    epsilons = np.logspace(np.log10(eps_lo), np.log10(eps_hi), n_eps)

    steps = [s["step"] for s in spectra]
    n_evals = len(spectra[0]["eigenvalues"])
    heatmap = np.zeros((n_eps, len(spectra)))

    for j, spec in enumerate(spectra):
        Z = build_dendrogram(spec["eigenvalues"])
        _, counts = cluster_count_curve(Z, n_evals, epsilon_range=epsilons)
        heatmap[:, j] = counts

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.pcolormesh(
        range(len(steps)), epsilons, heatmap,
        shading="nearest", cmap="viridis",
    )
    ax.set_yscale("log")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=45, fontsize=6)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Resolution ε")
    ax.set_title("Cluster Count n(ε) Over Training")
    plt.colorbar(im, ax=ax, label="Number of clusters")
    fig.tight_layout()
    return fig


def plot_summary_stats(spectra: list[dict]) -> Figure:
    """Summary statistics over training: loss, accuracy, spectral properties.

    Adapts labels for top-k Lanczos data (no negative eigenvalues) vs full spectra.
    """
    steps = [s["step"] for s in spectra]
    is_topk = not any(np.any(s["eigenvalues"] < 0) for s in spectra)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Training loss
    ax = axes[0, 0]
    losses = [s["train_loss"] for s in spectra]
    ax.plot(steps, losses, "o-", markersize=3)
    ax.set_ylabel("Training loss")
    ax.set_title("Training Loss")

    # Test accuracy
    ax = axes[0, 1]
    accs = [s["test_acc"] for s in spectra]
    ax.plot(steps, accs, "o-", markersize=3)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Test Accuracy")

    # Eigenvalue count / condition number
    ax = axes[0, 2]
    if is_topk:
        # With top-k only, show condition number (max/min of returned evals)
        cond = []
        for s in spectra:
            evals = s["eigenvalues"]
            pos = evals[evals > 0]
            cond.append(pos.max() / pos.min() if len(pos) > 1 else 1.0)
        ax.semilogy(steps, cond, "o-", markersize=3)
        ax.set_ylabel("λ_max / λ_min")
        ax.set_title("Top-k Condition Number")
    else:
        n_neg = [np.sum(s["eigenvalues"] < 0) for s in spectra]
        ax.plot(steps, n_neg, "o-", markersize=3)
        ax.set_ylabel("Count")
        ax.set_title("Negative Eigenvalues")

    # Spectral norm (max eigenvalue)
    ax = axes[1, 0]
    max_evals = [s["eigenvalues"].max() for s in spectra]
    min_evals = [s["eigenvalues"].min() for s in spectra]
    ax.plot(steps, max_evals, "o-", markersize=3, label="max λ")
    ax.plot(steps, min_evals, "o-", markersize=3, label="min λ")
    ax.legend()
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Spectral Range" + (" (top-k)" if is_topk else ""))

    # Sum of eigenvalues
    ax = axes[1, 1]
    traces = [s["eigenvalues"].sum() for s in spectra]
    ax.plot(steps, traces, "o-", markersize=3)
    ax.set_ylabel("Sum")
    ax.set_title("Top-k Eigenvalue Sum" if is_topk else "Hessian Trace")

    # Spectral entropy (of returned eigenvalues)
    ax = axes[1, 2]
    entropies = []
    for s in spectra:
        abs_evals = np.abs(s["eigenvalues"])
        total = abs_evals.sum()
        if total > 0:
            p = abs_evals / total
            p = p[p > 0]
            entropies.append(-np.sum(p * np.log(p)))
        else:
            entropies.append(0.0)
    ax.plot(steps, entropies, "o-", markersize=3)
    ax.set_ylabel("Entropy")
    ax.set_title("Top-k Spectral Entropy" if is_topk else "Spectral Entropy")

    for ax in axes.flat:
        ax.set_xlabel("Training step")

    fig.suptitle("Summary Statistics" + (" (Lanczos top-k)" if is_topk else ""),
                 fontsize=13)
    fig.tight_layout()
    return fig


def plot_conv_filters(weights, title: str = "Conv1 Filters") -> Figure:
    """Visualize first-layer convolution filters."""
    if hasattr(weights, "numpy"):
        weights = weights.detach().cpu().numpy()

    n_filters = weights.shape[0]
    cols = min(n_filters, 8)
    rows = (n_filters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    vmax = np.abs(weights).max()
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n_filters:
            ax.imshow(weights[i, 0], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"Filter {i}", fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


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

    n_panels = len(indices)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        r = results[idx]
        Z = r["linkage"]
        if Z.shape[0] > 0:
            dendrogram(Z, ax=ax, no_labels=True, color_threshold=0,
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
