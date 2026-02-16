# src/run_experiment.py
"""End-to-end pipeline: train, compute Hessian spectra, build dendrograms, plot."""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.model import LeNetTiny
from src.train import (
    get_mnist_loaders,
    get_hessian_subsample,
    train_with_checkpoints,
)
from src.hessian import compute_model_hessian, compute_eigenspectrum
from src.dendrogram import build_dendrogram, extract_top_gaps
from src.visualize import (
    plot_spectrum_evolution,
    plot_dendrogram_snapshots,
    plot_gap_barcode,
    plot_cluster_heatmap,
    plot_summary_stats,
    plot_conv_filters,
)


def run(
    n_epochs: int = 30,
    lr: float = 0.01,
    momentum: float = 0.9,
    batch_size: int = 128,
    hessian_n_samples: int = 1000,
    n_save_vectors: int = 50,
    seed: int = 42,
    results_dir: str = "results",
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

    model = LeNetTiny()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: LeNetTiny, {n_params} parameters")

    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    checkpoint_results = train_with_checkpoints(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=n_epochs,
        lr=lr,
        momentum=momentum,
        checkpoint_dir=checkpoint_dir,
    )

    # Save checkpoint metadata
    with open(results_dir / "checkpoints.json", "w") as f:
        json.dump(checkpoint_results, f, indent=2)
    print(f"Saved {len(checkpoint_results)} checkpoints")

    # ---- Phase 2: Compute Hessian spectra ----
    print("\n" + "=" * 60)
    print("Phase 2: Hessian Eigenspectra")
    print("=" * 60)

    hessian_data, hessian_targets = get_hessian_subsample(n_samples=hessian_n_samples, seed=seed)

    spectra = []
    for ckpt in tqdm(checkpoint_results, desc="Computing Hessian spectra"):
        # Load checkpoint
        state_dict = torch.load(ckpt["state_dict_path"], weights_only=True)
        model.load_state_dict(state_dict)

        # Compute Hessian
        H = compute_model_hessian(model, hessian_data, hessian_targets)

        # Eigendecompose
        eigenvalues, eigenvectors = compute_eigenspectrum(H, n_save_vectors=n_save_vectors)

        # Save spectrum
        spec_path = spectra_dir / f"step_{ckpt['step']:06d}.npz"
        np.savez(
            spec_path,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            step=ckpt["step"],
            epoch=ckpt["epoch"],
        )

        spectra.append({
            "step": ckpt["step"],
            "epoch": ckpt["epoch"],
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

    # Plot 1: Spectrum evolution
    fig = plot_spectrum_evolution(spectra)
    fig.savefig(figures_dir / "spectrum_evolution.png", dpi=150, bbox_inches="tight")
    print("  Saved spectrum_evolution.png")

    # Plot 2: Dendrogram snapshots
    fig = plot_dendrogram_snapshots(spectra)
    fig.savefig(figures_dir / "dendrogram_snapshots.png", dpi=150, bbox_inches="tight")
    print("  Saved dendrogram_snapshots.png")

    # Plot 3: Gap barcode
    fig = plot_gap_barcode(spectra, k=15)
    fig.savefig(figures_dir / "gap_barcode.png", dpi=150, bbox_inches="tight")
    print("  Saved gap_barcode.png")

    # Plot 4: Cluster count heatmap
    fig = plot_cluster_heatmap(spectra)
    fig.savefig(figures_dir / "cluster_heatmap.png", dpi=150, bbox_inches="tight")
    print("  Saved cluster_heatmap.png")

    # Plot 5: Summary stats
    fig = plot_summary_stats(spectra)
    fig.savefig(figures_dir / "summary_stats.png", dpi=150, bbox_inches="tight")
    print("  Saved summary_stats.png")

    # Plot 6: Conv filters at key epochs
    key_indices = [0, len(spectra) // 4, len(spectra) // 2, -1]
    for idx in key_indices:
        ckpt = checkpoint_results[idx]
        state_dict = torch.load(ckpt["state_dict_path"], weights_only=True)
        model.load_state_dict(state_dict)
        conv1_weight = model.features[0].weight
        fig = plot_conv_filters(conv1_weight, title=f"Conv1 Filters (step {ckpt['step']})")
        fig.savefig(figures_dir / f"filters_step_{ckpt['step']:06d}.png",
                    dpi=150, bbox_inches="tight")
    print("  Saved filter visualizations")

    # Close all figures to free memory
    import matplotlib.pyplot as plt
    plt.close("all")

    print("\n" + "=" * 60)
    print(f"Done! All results in {results_dir}/")
    print(f"  Figures: {figures_dir}/")
    print(f"  Spectra: {spectra_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    run()
