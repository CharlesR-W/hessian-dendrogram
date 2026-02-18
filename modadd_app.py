# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy>=1.24",
#     "plotly>=5.18",
#     "scipy>=1.10",
#     "matplotlib>=3.7",
#     "pillow>=10.0",
# ]
#
# [tool.marimo]
# width = "medium"
# theme = "dark"
# ///

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


# Cell 1: Imports
@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import json
    import io
    import base64
    from pathlib import Path
    from PIL import Image
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pio.templates.default = "plotly_dark"
    plt.style.use("dark_background")

    PALETTE = [
        "#7dd3fc", "#fcd34d", "#6ee7b7", "#fca5a5", "#94a3b8",
        "#c4b5fd", "#5eead4", "#f9a8d4", "#a5b4fc", "#fdba74",
    ]

    return (
        Image, PALETTE, Path, base64, fcluster, go, io, json, linkage,
        make_subplots, matplotlib, mo, np, pio, plt, scipy_dendrogram,
    )


# Cell 2: Data loading
@app.cell(hide_code=True)
def _(json, linkage, np, Path):
    _results_dir = Path("results_modadd")

    with open(_results_dir / "checkpoints.json") as _f:
        checkpoints = json.load(_f)

    spectra = []
    for _ckpt in checkpoints:
        _path = _results_dir / "spectra" / f"step_{_ckpt['step']:06d}.npz"
        _data = np.load(_path)
        spectra.append({
            "step": _ckpt["step"],
            "epoch": _ckpt.get("epoch", _ckpt["step"]),
            "train_loss": _ckpt["train_loss"],
            "train_acc": _ckpt.get("train_acc", 0.0),
            "test_acc": _ckpt["test_acc"],
            "eigenvalues": _data["eigenvalues"],
        })

    steps = [s["step"] for s in spectra]

    # Precompute all linkage matrices (used by multiple cells)
    linkage_matrices = []
    for _s in spectra:
        linkage_matrices.append(linkage(_s["eigenvalues"].reshape(-1, 1), method="single"))

    return checkpoints, linkage_matrices, spectra, steps


# Cell 3: Header markdown
@app.cell(hide_code=True)
def _(mo, spectra):
    mo.md(
        r"""
        # Modular Addition Grokking — Hessian Eigenspectrum Dendrograms

        Interactive exploration of how the Hessian eigenvalue spectrum — and its
        hierarchical cluster structure — evolve during **grokking** in a 1-layer
        transformer trained on $a + b \pmod{p}$.

        **Core idea:** At resolution $\varepsilon$, eigenvalues within $\varepsilon$ of
        each other are "coarse-grained" into the same block.  This is equivalent to
        single-linkage clustering on the sorted eigenvalue array.  The resulting
        dendrogram reveals hierarchical spectral structure that changes as the network
        memorizes, then generalizes.
        """
        + f"\n\n**Loaded {len(spectra)} checkpoints** — "
        f"{len(spectra[0]['eigenvalues']):,} eigenvalues each — "
        f"step {spectra[0]['step']} to {spectra[-1]['step']} — "
        f"final test acc {spectra[-1]['test_acc']:.1%}."
    )
    return


# Cell 4: Grokking Curve header
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Grokking Curve

        The hallmark of grokking: training accuracy reaches 100% early, but
        test accuracy remains at chance until a delayed phase transition.
        The gap between memorization and generalization is clearly visible.
        """
    )
    return


# Cell 5: Grokking Curve plot
@app.cell(hide_code=True)
def _(go, make_subplots, mo, spectra, steps):
    _fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Train loss on left y-axis (log scale)
    _fig.add_trace(
        go.Scatter(
            x=steps,
            y=[_s["train_loss"] for _s in spectra],
            mode="lines+markers",
            name="Train Loss",
            line=dict(color="#7dd3fc", width=2),
            marker=dict(size=4),
            hovertemplate="Step %{x}<br>Train Loss: %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Train accuracy on right y-axis (linear 0-1)
    _fig.add_trace(
        go.Scatter(
            x=steps,
            y=[_s["train_acc"] for _s in spectra],
            mode="lines+markers",
            name="Train Acc",
            line=dict(color="#6ee7b7", width=2, dash="solid"),
            marker=dict(size=4),
            hovertemplate="Step %{x}<br>Train Acc: %{y:.3f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Test accuracy on right y-axis (dashed)
    _fig.add_trace(
        go.Scatter(
            x=steps,
            y=[_s["test_acc"] for _s in spectra],
            mode="lines+markers",
            name="Test Acc",
            line=dict(color="#fca5a5", width=2, dash="dash"),
            marker=dict(size=4),
            hovertemplate="Step %{x}<br>Test Acc: %{y:.3f}<extra></extra>",
        ),
        secondary_y=True,
    )

    _fig.update_layout(
        title="Grokking Dynamics: Loss and Accuracy Over Training",
        height=480,
        legend=dict(font=dict(size=11)),
    )
    _fig.update_xaxes(title_text="Training Step")
    _fig.update_yaxes(title_text="Train Loss", type="log", secondary_y=False)
    _fig.update_yaxes(
        title_text="Accuracy", range=[0, 1.05], secondary_y=True,
    )

    grokking_fig = mo.ui.plotly(_fig)
    grokking_fig
    return (grokking_fig,)


# Cell 6: Spectrum Evolution header
@app.cell(hide_code=True)
def _(mo):
    mo.md("## Spectrum Evolution")
    return


# Cell 7: Spectrum Evolution plot
@app.cell(hide_code=True)
def _(PALETTE, go, mo, np, spectra):
    _fig = go.Figure()
    for _i, _s in enumerate(spectra):
        _evals = np.sort(_s["eigenvalues"])
        _pos = _evals[_evals > 0]
        _color = PALETTE[_i % len(PALETTE)]
        _opacity = 0.3 + 0.7 * (_i / max(len(spectra) - 1, 1))
        _fig.add_trace(go.Scatter(
            x=np.arange(len(_pos)),
            y=_pos,
            mode="lines",
            name=f"Step {_s['step']}",
            line=dict(color=_color, width=1.5),
            opacity=_opacity,
            hovertemplate=(
                f"Step {_s['step']} (ep {_s['epoch']})<br>"
                "Index: %{x}<br>λ: %{y:.4f}<extra></extra>"
            ),
        ))

    _fig.update_layout(
        title="Positive Eigenvalue Spectrum Over Training",
        xaxis_title="Index (sorted)",
        yaxis_title="Eigenvalue",
        yaxis_type="log",
        height=480,
        legend=dict(font=dict(size=10)),
    )

    spectrum_fig = mo.ui.plotly(_fig)
    spectrum_fig
    return (spectrum_fig,)


# Cell 8: Interactive Dendrogram header
@app.cell(hide_code=True)
def _(mo, spectra):
    mo.md(
        r"""
        ## Interactive Dendrogram (Truncated)

        With """
        + f"{len(spectra[0]['eigenvalues']):,}"
        + r""" eigenvalues, the full dendrogram is unreadable.  This shows only
        the **top 30 merges** — the coarsest spectral structure.  Each leaf represents
        a cluster of eigenvalues; the number in parentheses is the cluster size.
        Taller branches = larger spectral gaps.
        """
    )
    return


# Cell 9: Slider
@app.cell(hide_code=True)
def _(mo, spectra):
    step_slider = mo.ui.slider(
        start=0,
        stop=len(spectra) - 1,
        value=len(spectra) // 2,
        label="Checkpoint index",
        show_value=True,
        full_width=True,
    )
    step_slider
    return (step_slider,)


# Cell 10: Interactive Dendrogram plot
@app.cell(hide_code=True)
def _(linkage_matrices, mo, np, plt, scipy_dendrogram, spectra, step_slider):
    _idx = step_slider.value
    _s = spectra[_idx]
    _evals = _s["eigenvalues"]
    _Z = linkage_matrices[_idx]

    _mpl_fig, _ax = plt.subplots(figsize=(14, 6), facecolor="#111111")
    _ax.set_facecolor("#111111")

    scipy_dendrogram(
        _Z, ax=_ax, truncate_mode="lastp", p=30,
        show_leaf_counts=True, no_labels=False,
        color_threshold=0, above_threshold_color="#7dd3fc",
        leaf_font_size=9,
    )

    _ax.set_ylabel("Gap size (ε)", fontsize=12, color="#e2e8f0")
    _ax.set_title(
        f"Step {_s['step']} (epoch {_s['epoch']})  —  "
        f"loss={_s['train_loss']:.4f}, train_acc={_s['train_acc']:.1%}, "
        f"test_acc={_s['test_acc']:.1%}",
        fontsize=14, color="#e2e8f0",
    )
    _ax.tick_params(colors="#94a3b8")
    _ax.spines["bottom"].set_color("#334155")
    _ax.spines["left"].set_color("#334155")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _mpl_fig.tight_layout()

    _n_neg = int(np.sum(_evals < 0))
    _n_pos = int(np.sum(_evals > 0))
    _sorted = np.sort(_evals)
    _gaps = np.diff(_sorted)
    _top_gaps = np.sort(_gaps)[::-1][:10]

    dendro_plot = mo.vstack([
        _mpl_fig,
        mo.md(
            f"**{_n_neg:,}** negative · **{_n_pos:,}** positive "
            f"&nbsp;|&nbsp; Range: [{_evals.min():.3f}, {_evals.max():.3f}] "
            f"&nbsp;|&nbsp; Top 3 gaps: {_top_gaps[0]:.3f}, {_top_gaps[1]:.3f}, {_top_gaps[2]:.3f}"
        ),
    ])
    plt.close(_mpl_fig)
    dendro_plot
    return (dendro_plot,)


# Cell 11: Dendrogram Evolution header
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Dendrogram Evolution (Animated)

        How does the spectral hierarchy change over training?  Each frame shows the
        truncated dendrogram at one checkpoint, with a fixed y-axis for comparison.
        Watch for the hierarchical structure to emerge and reorganize as the network
        transitions from memorization to generalization.
        """
    )
    return


# Cell 12: Dendrogram Evolution GIF
@app.cell(hide_code=True)
def _(
    Image, base64, io, linkage_matrices, mo, np, plt,
    scipy_dendrogram, spectra,
):
    # Find global y-axis max for consistent scaling
    _ymax = max(_Z[:, 2].max() for _Z in linkage_matrices) * 1.1

    _frames = []
    for _i, _s in enumerate(spectra):
        _Z = linkage_matrices[_i]
        _fig, _ax = plt.subplots(figsize=(12, 5), facecolor="#111111")
        _ax.set_facecolor("#111111")

        scipy_dendrogram(
            _Z, ax=_ax, truncate_mode="lastp", p=30,
            show_leaf_counts=True, no_labels=False,
            color_threshold=0, above_threshold_color="#7dd3fc",
            leaf_font_size=8,
        )

        _ax.set_ylim(0, _ymax)
        _ax.set_ylabel("Gap size (ε)", fontsize=11, color="#e2e8f0")
        _ax.set_title(
            f"Step {_s['step']}  (epoch {_s['epoch']})  —  "
            f"loss={_s['train_loss']:.4f}  train_acc={_s['train_acc']:.1%}  "
            f"test_acc={_s['test_acc']:.1%}",
            fontsize=13, color="#e2e8f0",
        )
        _ax.tick_params(colors="#94a3b8")
        _ax.spines["bottom"].set_color("#334155")
        _ax.spines["left"].set_color("#334155")
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _fig.tight_layout()

        _buf = io.BytesIO()
        _fig.savefig(_buf, format="png", dpi=100, facecolor="#111111")
        _buf.seek(0)
        _frames.append(Image.open(_buf).copy())
        plt.close(_fig)
        _buf.close()

    # Build GIF: hold first and last frames longer
    _durations = [1200] + [600] * (len(_frames) - 2) + [1200]
    _gif_buf = io.BytesIO()
    _frames[0].save(
        _gif_buf, format="GIF", save_all=True,
        append_images=_frames[1:],
        duration=_durations, loop=0,
    )
    _gif_buf.seek(0)
    _b64 = base64.b64encode(_gif_buf.read()).decode()

    gif_display = mo.Html(
        f'<img src="data:image/gif;base64,{_b64}" '
        f'style="width:100%; border-radius:8px;" />'
    )
    gif_display
    return (gif_display,)


# Cell 13: Cluster Count Heatmap header
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Cluster Count $n(\varepsilon)$ — Log Scale

        How many spectral clusters exist at each resolution $\varepsilon$?
        Color is **log₁₀(clusters)** for better contrast.  Dark purple = few
        clusters (eigenvalues merged at coarse resolution), bright yellow = many
        clusters (fine-grained).
        """
    )
    return


# Cell 14: Cluster Count Heatmap plot
@app.cell(hide_code=True)
def _(fcluster, go, linkage_matrices, mo, np, spectra):
    _n_params = len(spectra[0]["eigenvalues"])
    _eps_range = np.logspace(-12, 2, 100)
    _heatmap = np.zeros((len(_eps_range), len(spectra)))

    for _j, _s in enumerate(spectra):
        _Z = linkage_matrices[_j]
        for _i, _eps in enumerate(_eps_range):
            _labels = fcluster(_Z, t=_eps, criterion="distance")
            _heatmap[_i, _j] = len(set(_labels))

    _log_heatmap = np.log10(_heatmap + 1)

    _fig = go.Figure(data=go.Heatmap(
        z=_log_heatmap,
        x=[f"{_s['step']}" for _s in spectra],
        y=[f"{_e:.1e}" for _e in _eps_range],
        colorscale="Viridis",
        colorbar=dict(
            title="log₁₀(clusters)",
            tickvals=[0, 1, 2, 3, np.log10(_n_params)],
            ticktext=["1", "10", "100", "1k", f"{_n_params}"],
        ),
        hovertemplate=(
            "Step %{x}<br>ε = %{y}<br>"
            "Clusters: %{customdata}<extra></extra>"
        ),
        customdata=_heatmap.astype(int),
    ))
    _fig.update_layout(
        title="Cluster Count n(ε) Over Training (log scale)",
        xaxis_title="Training step",
        yaxis_title="Resolution ε",
        height=500,
    )

    heatmap_fig = mo.ui.plotly(_fig)
    heatmap_fig
    return (heatmap_fig,)


# Cell 15: n(ε) Curves header
@app.cell(hide_code=True)
def _(mo):
    mo.md("## n(ε) Curves — Selected Checkpoints")
    return


# Cell 16: n(ε) Curves plot
@app.cell(hide_code=True)
def _(PALETTE, fcluster, go, linkage_matrices, mo, np, spectra):
    _indices = sorted(set([
        0, 1, 4,
        len(spectra) // 4,
        len(spectra) // 2,
        len(spectra) - 1,
    ]))

    _fig = go.Figure()
    for _ci, _idx in enumerate(_indices):
        _s = spectra[_idx]
        _Z = linkage_matrices[_idx]
        _merge_heights = _Z[:, 2]
        _lo = max(_merge_heights.min() * 0.5, 1e-12)
        _hi = _merge_heights.max() * 2.0
        _eps_arr = np.logspace(np.log10(_lo), np.log10(_hi), 200)
        _counts = np.array([
            len(set(fcluster(_Z, t=_e, criterion="distance")))
            for _e in _eps_arr
        ])
        _fig.add_trace(go.Scatter(
            x=_eps_arr,
            y=_counts,
            mode="lines",
            name=f"Step {_s['step']} (ep {_s['epoch']})",
            line=dict(color=PALETTE[_ci % len(PALETTE)], width=2),
        ))

    _fig.update_layout(
        title="n(ε): Number of Clusters vs Resolution",
        xaxis_title="ε",
        yaxis_title="Number of clusters",
        xaxis_type="log",
        yaxis_type="log",
        height=450,
    )

    neps_fig = mo.ui.plotly(_fig)
    neps_fig
    return (neps_fig,)


# Cell 17: Top Spectral Gaps header
@app.cell(hide_code=True)
def _(mo):
    mo.md("## Top Spectral Gaps Over Training")
    return


# Cell 18: Top Spectral Gaps plot
@app.cell(hide_code=True)
def _(PALETTE, go, mo, np, spectra, steps):
    _k = 10
    _gap_matrix = np.zeros((len(spectra), _k))
    for _j, _s in enumerate(spectra):
        _sorted_evals = np.sort(_s["eigenvalues"])
        _gaps = np.diff(_sorted_evals)
        _top = np.sort(_gaps)[::-1][:_k]
        _gap_matrix[_j, :len(_top)] = _top

    _fig = go.Figure()
    for _g in range(_k):
        _fig.add_trace(go.Scatter(
            x=steps,
            y=_gap_matrix[:, _g],
            mode="lines+markers",
            name=f"Gap {_g + 1}",
            line=dict(color=PALETTE[_g % len(PALETTE)], width=1.5),
            marker=dict(size=4),
        ))

    _fig.update_layout(
        title=f"Top-{_k} Spectral Gaps Over Training",
        xaxis_title="Training step",
        yaxis_title="Gap size",
        yaxis_type="log",
        height=450,
    )

    gaps_fig = mo.ui.plotly(_fig)
    gaps_fig
    return (gaps_fig,)


# Cell 19: Summary Statistics header
@app.cell(hide_code=True)
def _(mo):
    mo.md("## Summary Statistics")
    return


# Cell 20: Summary Statistics plot
@app.cell(hide_code=True)
def _(go, make_subplots, mo, np, spectra, steps):
    _train_losses = [_s["train_loss"] for _s in spectra]
    _train_accs = [_s["train_acc"] for _s in spectra]
    _test_accs = [_s["test_acc"] for _s in spectra]
    _max_evals = [float(np.max(_s["eigenvalues"])) for _s in spectra]
    _traces = [float(np.sum(_s["eigenvalues"])) for _s in spectra]

    _entropies = []
    for _s in spectra:
        _abs = np.abs(_s["eigenvalues"])
        _abs = _abs[_abs > 0]
        _p = _abs / _abs.sum()
        _entropies.append(float(-np.sum(_p * np.log(_p))))

    _fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Train Loss", "Train Accuracy", "Test Accuracy",
            "Max Eigenvalue", "Hessian Trace", "Spectral Entropy",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    _colors = ["#7dd3fc", "#6ee7b7", "#fca5a5", "#fcd34d", "#c4b5fd", "#5eead4"]
    _data = [_train_losses, _train_accs, _test_accs, _max_evals, _traces, _entropies]
    for _i, (_d, _c) in enumerate(zip(_data, _colors)):
        _fig.add_trace(go.Scatter(
            x=steps, y=_d, mode="lines+markers",
            line=dict(color=_c), marker=dict(size=4),
            showlegend=False,
        ), row=_i // 3 + 1, col=_i % 3 + 1)

    _fig.update_layout(height=550, title_text="Summary Statistics Over Training")
    # Log scale for loss and trace
    _fig.update_yaxes(type="log", row=1, col=1)
    _fig.update_yaxes(type="log", row=2, col=2)
    for _i in range(6):
        _fig.update_xaxes(title_text="Step", row=_i // 3 + 1, col=_i % 3 + 1)

    summary_fig = mo.ui.plotly(_fig)
    summary_fig
    return (summary_fig,)


# Cell 21: Eigenvalue Histogram header
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Eigenvalue Distribution

        Histogram of eigenvalues at the selected checkpoint, split positive/negative.
        Use the slider above to change checkpoint.
        """
    )
    return


# Cell 22: Eigenvalue Histogram plot
@app.cell(hide_code=True)
def _(go, mo, np, spectra, step_slider):
    _idx = step_slider.value
    _s = spectra[_idx]
    _evals = _s["eigenvalues"]

    _fig = go.Figure()
    _fig.add_trace(go.Histogram(
        x=_evals[_evals < 0], nbinsx=80,
        name="Negative", marker_color="#fca5a5", opacity=0.7,
    ))
    _fig.add_trace(go.Histogram(
        x=_evals[_evals >= 0], nbinsx=80,
        name="Positive", marker_color="#7dd3fc", opacity=0.7,
    ))
    _fig.update_layout(
        title=f"Eigenvalue Distribution — Step {_s['step']} (epoch {_s['epoch']})",
        xaxis_title="Eigenvalue",
        yaxis_title="Count",
        barmode="overlay",
        height=400,
    )

    hist_fig = mo.ui.plotly(_fig)
    hist_fig
    return (hist_fig,)


# Cell 23: Footer
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        *Modular Addition Grokking — 1-layer transformer, d_model=32,
        $a + b \pmod{113}$, full Hessian eigendecomposition at checkpoints
        throughout training.  The grokking transition (delayed generalization
        after memorization) is reflected in structural changes of the Hessian
        eigenspectrum.*
        """
    )
    return


if __name__ == "__main__":
    app.run()
