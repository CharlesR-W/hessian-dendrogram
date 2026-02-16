# tests/test_visualize.py
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
from pathlib import Path
from src.visualize import (
    plot_spectrum_evolution,
    plot_dendrogram_snapshots,
    plot_gap_barcode,
    plot_cluster_heatmap,
    plot_summary_stats,
    plot_conv_filters,
)


def _make_fake_spectra(n_checkpoints=5, n_params=100):
    """Generate fake eigenvalue data for testing plots."""
    spectra = []
    for i in range(n_checkpoints):
        # Progressively more structured spectrum
        evals = np.random.randn(n_params) * (1.0 + i * 0.5)
        evals.sort()
        spectra.append({
            "step": i * 100,
            "epoch": i,
            "eigenvalues": evals,
            "train_loss": 2.0 - i * 0.3,
            "test_acc": 0.1 + i * 0.15,
        })
    return spectra


def test_plot_spectrum_evolution(tmp_path):
    spectra = _make_fake_spectra()
    fig = plot_spectrum_evolution(spectra)
    fig.savefig(tmp_path / "test_spectrum.png")
    assert (tmp_path / "test_spectrum.png").exists()


def test_plot_dendrogram_snapshots(tmp_path):
    spectra = _make_fake_spectra()
    fig = plot_dendrogram_snapshots(spectra, indices=[0, 2, 4])
    fig.savefig(tmp_path / "test_dendrograms.png")
    assert (tmp_path / "test_dendrograms.png").exists()


def test_plot_cluster_heatmap(tmp_path):
    spectra = _make_fake_spectra()
    fig = plot_cluster_heatmap(spectra)
    fig.savefig(tmp_path / "test_heatmap.png")
    assert (tmp_path / "test_heatmap.png").exists()


def test_plot_summary_stats(tmp_path):
    spectra = _make_fake_spectra()
    fig = plot_summary_stats(spectra)
    fig.savefig(tmp_path / "test_summary.png")
    assert (tmp_path / "test_summary.png").exists()


def test_plot_conv_filters(tmp_path):
    import torch
    # Fake conv weights: 8 filters, 1 input channel, 5x5
    weights = torch.randn(8, 1, 5, 5)
    fig = plot_conv_filters(weights, title="Test filters")
    fig.savefig(tmp_path / "test_filters.png")
    assert (tmp_path / "test_filters.png").exists()
