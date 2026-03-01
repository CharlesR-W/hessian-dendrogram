"""Pipeline: load cached Hessians, compute Fiedler dendrograms, visualize.

Usage:
    cd /home/crw/Programming/Claude/hessian-dendrogram
    uv run python -m src.run_fiedler [--results-dir results]
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
