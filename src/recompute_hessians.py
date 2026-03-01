"""Recompute and cache full Hessian matrices from saved checkpoints.

Usage:
    cd /home/crw/Programming/Claude/hessian-dendrogram
    uv run python -m src.recompute_hessians [--results-dir results] [--n-samples 50]
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
