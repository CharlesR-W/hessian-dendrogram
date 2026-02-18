"""End-to-end pipeline for modular addition grokking + Hessian analysis."""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.modadd_model import ModAddTransformer
from src.modadd_train import make_modadd_data, train_modadd
from src.hessian import compute_model_hessian, compute_eigenspectrum, compute_lanczos_eigenspectrum
from src.visualize import (
    plot_spectrum_evolution,
    plot_dendrogram_snapshots,
    plot_gap_barcode,
    plot_cluster_heatmap,
    plot_summary_stats,
)


def run(
    p: int = 113,
    d_model: int = 128,
    n_heads: int = 4,
    n_steps: int = 50000,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    train_fraction: float = 0.3,
    hessian_n_samples: int = 200,
    lanczos_k: int = 200,
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

    n_params = sum(p_.numel() for p_ in model.parameters())
    use_lanczos = n_params > 20000
    if use_lanczos:
        print(f"Using Lanczos (k={lanczos_k}) — model has {n_params} params")
    else:
        print(f"Using full Hessian — model has {n_params} params")

    spectra = []
    for ckpt in tqdm(checkpoint_results, desc="Computing Hessian spectra"):
        state_dict = torch.load(ckpt["state_dict_path"], weights_only=True)
        model.load_state_dict(state_dict)

        if use_lanczos:
            eigenvalues, eigenvectors = compute_lanczos_eigenspectrum(
                model, hessian_data, hessian_targets,
                k=lanczos_k, n_save_vectors=n_save_vectors,
            )
        else:
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
        print(f"  Step {ckpt['step']:6d}: {len(eigenvalues)} eigenvalues, "
              f"{n_neg} negative, range [{eigenvalues.min():.2f}, {eigenvalues.max():.2f}]")

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
