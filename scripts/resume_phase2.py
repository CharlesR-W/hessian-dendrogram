"""Resume Phase 2 (Lanczos Hessian) + run Phase 3 (visualization).

Skips checkpoints that already have spectra computed.
"""
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.modadd_model import ModAddTransformer
from src.modadd_train import make_modadd_data
from src.hessian import compute_lanczos_eigenspectrum
from src.visualize import (
    plot_spectrum_evolution,
    plot_dendrogram_snapshots,
    plot_gap_barcode,
    plot_cluster_heatmap,
    plot_summary_stats,
)

results_dir = Path("results_modadd")
spectra_dir = results_dir / "spectra"
figures_dir = results_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Load checkpoint metadata
with open(results_dir / "checkpoints.json") as f:
    checkpoint_results = json.load(f)

print(f"Total checkpoints: {len(checkpoint_results)}")

# Find which spectra already exist
existing = {p.stem for p in spectra_dir.glob("*.npz")}
remaining = [c for c in checkpoint_results if f"step_{c['step']:06d}" not in existing]
print(f"Already computed: {len(existing)}, remaining: {len(remaining)}")

# Set up model and data
p, d_model, n_heads = 113, 128, 4
model = ModAddTransformer(p=p, d_model=d_model, n_heads=n_heads)
train_tok, train_lab, _, _ = make_modadd_data(p=p, train_fraction=0.3, seed=42)

rng = torch.Generator().manual_seed(43)
n_sub = min(200, len(train_tok))
sub_idx = torch.randperm(len(train_tok), generator=rng)[:n_sub]
hessian_data = train_tok[sub_idx]
hessian_targets = train_lab[sub_idx]
print(f"Hessian subsample: {n_sub} training pairs")

# Phase 2: compute remaining spectra
lanczos_k = 200
n_save_vectors = 50

for ckpt in tqdm(remaining, desc="Computing Hessian spectra"):
    state_dict = torch.load(ckpt["state_dict_path"], weights_only=True)
    model.load_state_dict(state_dict)

    eigenvalues, eigenvectors = compute_lanczos_eigenspectrum(
        model, hessian_data, hessian_targets,
        k=lanczos_k, n_save_vectors=n_save_vectors,
    )

    spec_path = spectra_dir / f"step_{ckpt['step']:06d}.npz"
    np.savez(
        spec_path,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        step=ckpt["step"],
    )

    n_neg = np.sum(eigenvalues < 0)
    tqdm.write(
        f"  Step {ckpt['step']:6d}: {len(eigenvalues)} eigenvalues, "
        f"{n_neg} negative, range [{eigenvalues.min():.2f}, {eigenvalues.max():.2f}]"
    )

# Phase 3: generate visualizations from ALL spectra
print("\n" + "=" * 60)
print("Phase 3: Visualization")
print("=" * 60)

spectra = []
for ckpt in checkpoint_results:
    spec_path = spectra_dir / f"step_{ckpt['step']:06d}.npz"
    data = np.load(spec_path)
    spectra.append({
        "step": ckpt["step"],
        "train_loss": ckpt["train_loss"],
        "test_acc": ckpt["test_acc"],
        "eigenvalues": data["eigenvalues"],
    })

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

print(f"\nDone! All results in {results_dir}/")
