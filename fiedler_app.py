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


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import json
    from pathlib import Path
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram, fcluster
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pio.templates.default = "plotly_dark"
    plt.style.use("dark_background")

    PALETTE = [
        "#7dd3fc", "#fcd34d", "#6ee7b7", "#fca5a5", "#94a3b8",
        "#c4b5fd", "#5eead4", "#f9a8d4", "#a5b4fc", "#fdba74",
    ]

    LAYER_COLORS = {
        "features.0": "#7dd3fc",
        "features.3": "#fcd34d",
        "classifier": "#6ee7b7",
    }

    return (
        LAYER_COLORS, PALETTE, Path, fcluster, go, json,
        make_subplots, matplotlib, mo, np, pio, plt, scipy_dendrogram,
    )


@app.cell(hide_code=True)
def _(json, np, Path):
    _results_dir = Path("results")

    with open(_results_dir / "checkpoints.json") as _f:
        checkpoints = json.load(_f)

    # Load Fiedler dendrograms
    fiedler_data = []
    for _ckpt in checkpoints:
        _path = _results_dir / "fiedler" / f"step_{_ckpt['step']:06d}.npz"
        _d = np.load(_path)
        fiedler_data.append({
            "step": _ckpt["step"],
            "epoch": _ckpt["epoch"],
            "train_loss": _ckpt["train_loss"],
            "test_acc": _ckpt["test_acc"],
            "linkage": _d["linkage"],
        })

    # Load eigenvalue spectra (for comparison)
    spectra = []
    for _ckpt in checkpoints:
        _path = _results_dir / "spectra" / f"step_{_ckpt['step']:06d}.npz"
        _d = np.load(_path)
        spectra.append({
            "step": _ckpt["step"],
            "epoch": _ckpt["epoch"],
            "eigenvalues": _d["eigenvalues"],
        })

    steps = [_fd["step"] for _fd in fiedler_data]

    # Compute parameter-to-layer labels
    # LeNetTiny: features.0 (conv1) = 208, features.3 (conv2) = 3216, classifier = 2570
    _layer_boundaries = [
        ("features.0", 208),
        ("features.3", 3216),
        ("classifier", 2570),
    ]
    param_labels = []
    for _name, _count in _layer_boundaries:
        param_labels.extend([_name] * _count)

    return checkpoints, fiedler_data, param_labels, spectra, steps


@app.cell(hide_code=True)
def _(fiedler_data, mo):
    mo.md(
        r"""
        # Fiedler Parameter Dendrogram Analysis

        Interactive exploration of how the **parameter-level coupling structure** of
        a LeNet-tiny (5,994 params) evolves during MNIST training.

        **Core idea:**  The full Hessian $H$ encodes second-order interactions between
        every pair of parameters.  We treat $|H_{ij}|$ as edge weights in a coupling
        graph, compute its graph Laplacian $L = D - A$, and recursively bipartition
        using the **Fiedler vector** (eigenvector of $\lambda_2$, the algebraic
        connectivity).  Parameters that merge late are *weakly coupled* in the loss
        landscape; parameters that merge early are *strongly coupled*.

        Unlike the eigenvalue-spectrum dendrogram (which clusters eigenvalues by
        spectral gaps), this dendrogram clusters **parameters** by their pairwise
        Hessian coupling strength.
        """
        + f"\n\n**Loaded {len(fiedler_data)} checkpoints** — "
        f"5,994 parameters each — "
        f"{fiedler_data[-1]['epoch']} epochs — "
        f"{fiedler_data[-1]['test_acc']:.1%} final accuracy."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Algebraic Connectivity Over Training")
    return


@app.cell(hide_code=True)
def _(fiedler_data, go, make_subplots, mo, np, steps):
    # Extract algebraic connectivity: lambda2 = 1 / root_merge_height
    _lambda2s = []
    for _fd in fiedler_data:
        _root_h = _fd["linkage"][-1, 2]
        _lam2 = 1.0 / _root_h if _root_h > 1e-15 else 0.0
        _lambda2s.append(_lam2)

    _losses = [_fd["train_loss"] for _fd in fiedler_data]
    _accs = [_fd["test_acc"] for _fd in fiedler_data]

    _fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Lambda2 on primary y-axis (log scale)
    _fig.add_trace(
        go.Scatter(
            x=steps, y=_lambda2s,
            mode="lines+markers",
            name="lambda_2 (algebraic connectivity)",
            line=dict(color="#7dd3fc", width=2.5),
            marker=dict(size=5),
            hovertemplate="Step %{x}<br>lambda_2: %{y:.3e}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Train loss on secondary y-axis
    _valid_loss_steps = [_s for _s, _l in zip(steps, _losses) if not np.isnan(_l)]
    _valid_losses = [_l for _l in _losses if not np.isnan(_l)]
    _fig.add_trace(
        go.Scatter(
            x=_valid_loss_steps, y=_valid_losses,
            mode="lines+markers",
            name="Train loss",
            line=dict(color="#fca5a5", width=1.5, dash="dash"),
            marker=dict(size=3),
            opacity=0.7,
            hovertemplate="Step %{x}<br>Loss: %{y:.4f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Test accuracy on secondary y-axis
    _fig.add_trace(
        go.Scatter(
            x=steps, y=_accs,
            mode="lines+markers",
            name="Test accuracy",
            line=dict(color="#6ee7b7", width=1.5, dash="dot"),
            marker=dict(size=3),
            opacity=0.7,
            hovertemplate="Step %{x}<br>Acc: %{y:.1%}<extra></extra>",
        ),
        secondary_y=True,
    )

    _fig.update_layout(
        title="Algebraic Connectivity (lambda_2) vs Training Progress",
        height=500,
        legend=dict(x=0.02, y=0.98, font=dict(size=11)),
    )
    _fig.update_xaxes(title_text="Training step")
    _fig.update_yaxes(
        title_text="lambda_2 (algebraic connectivity)",
        type="log",
        secondary_y=False,
    )
    _fig.update_yaxes(
        title_text="Loss / Accuracy",
        type="log",
        secondary_y=True,
    )

    connectivity_fig = mo.ui.plotly(_fig)
    connectivity_fig
    return (connectivity_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Interactive Truncated Fiedler Dendrogram

        The full dendrogram has 5,994 leaves (one per parameter) and is unreadable.
        This shows only the **top 30 merges** — the coarsest coupling structure.
        Each leaf represents a cluster of parameters; the number in parentheses is
        the cluster size.  Taller branches = weaker inter-cluster coupling (higher
        $1/\lambda_2$).
        """
    )
    return


@app.cell(hide_code=True)
def _(fiedler_data, mo):
    fiedler_step_slider = mo.ui.slider(
        start=0,
        stop=len(fiedler_data) - 1,
        value=len(fiedler_data) // 2,
        label="Checkpoint index",
        show_value=True,
        full_width=True,
    )
    fiedler_step_slider
    return (fiedler_step_slider,)


@app.cell(hide_code=True)
def _(fiedler_data, fiedler_step_slider, mo, np, plt, scipy_dendrogram):
    _idx = fiedler_step_slider.value
    _fd = fiedler_data[_idx]
    _Z = _fd["linkage"]

    _mpl_fig, _ax = plt.subplots(figsize=(14, 6), facecolor="#111111")
    _ax.set_facecolor("#111111")

    scipy_dendrogram(
        _Z, ax=_ax, truncate_mode="lastp", p=30,
        show_leaf_counts=True, no_labels=False,
        color_threshold=0, above_threshold_color="#7dd3fc",
        leaf_font_size=9,
    )

    _ax.set_ylabel("Merge height (1/lambda_2)", fontsize=12, color="#e2e8f0")
    _ax.set_title(
        f"Fiedler Dendrogram — Step {_fd['step']} (epoch {_fd['epoch']})  —  "
        f"loss={_fd['train_loss']:.4f}, acc={_fd['test_acc']:.1%}",
        fontsize=14, color="#e2e8f0",
    )
    _ax.tick_params(colors="#94a3b8")
    _ax.spines["bottom"].set_color("#334155")
    _ax.spines["left"].set_color("#334155")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _mpl_fig.tight_layout()

    # Merge height statistics
    _heights = _Z[:, 2]
    _top_heights = np.sort(_heights)[::-1][:10]

    _dendro_out = mo.vstack([
        _mpl_fig,
        mo.md(
            f"**Merge height range:** [{_heights.min():.2f}, {_heights.max():.2e}] "
            f"&nbsp;|&nbsp; **Median:** {np.median(_heights):.2f} "
            f"&nbsp;|&nbsp; **Top 3 heights:** {_top_heights[0]:.2e}, "
            f"{_top_heights[1]:.2e}, {_top_heights[2]:.2e}"
        ),
    ])
    plt.close(_mpl_fig)
    _dendro_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Layer Composition of Clusters

        Cut the Fiedler dendrogram into $k$ clusters and count which layer each
        cluster's parameters belong to.  If the Hessian coupling respects layer
        boundaries, each cluster should be dominated by a single layer.  Mixed
        clusters indicate cross-layer parameter coupling.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    cluster_k_slider = mo.ui.slider(
        start=3,
        stop=10,
        value=5,
        label="Number of clusters (k)",
        show_value=True,
        full_width=True,
    )
    cluster_k_slider
    return (cluster_k_slider,)


@app.cell(hide_code=True)
def _(
    LAYER_COLORS, cluster_k_slider, fcluster, fiedler_data,
    go, mo, np, param_labels, steps,
):
    _k = cluster_k_slider.value
    _layer_names = ["features.0", "features.3", "classifier"]
    _labels_arr = np.array(param_labels)

    # For each checkpoint, cut into k clusters and compute layer fractions
    _layer_fracs = {_ln: [] for _ln in _layer_names}

    for _fd in fiedler_data:
        _Z = _fd["linkage"]
        _cluster_ids = fcluster(_Z, t=_k, criterion="maxclust")

        # Count total params per layer across all clusters
        _total = len(_cluster_ids)
        for _ln in _layer_names:
            _mask = _labels_arr == _ln
            _frac = np.sum(_mask) / _total
            _layer_fracs[_ln].append(_frac)

    # Actually, the interesting thing is how clusters MIX layers.
    # Let's show per-cluster layer composition instead.
    # For each checkpoint, for each cluster, what fraction of its params
    # come from each layer?  Show as stacked bar per checkpoint.

    # Better approach: for each checkpoint, show the fraction of params
    # in "pure" clusters (>90% single-layer) vs "mixed" clusters
    _pure_fracs = []
    _mixed_fracs = []
    _per_layer_in_mixed = {_ln: [] for _ln in _layer_names}

    for _fd in fiedler_data:
        _Z = _fd["linkage"]
        _cluster_ids = fcluster(_Z, t=_k, criterion="maxclust")
        _total = len(_cluster_ids)
        _pure_count = 0

        for _cid in range(1, _k + 1):
            _cmask = _cluster_ids == _cid
            _csize = np.sum(_cmask)
            if _csize == 0:
                continue
            # Check layer composition of this cluster
            _layer_counts = {}
            for _ln in _layer_names:
                _layer_counts[_ln] = np.sum(_cmask & (_labels_arr == _ln))
            _dominant_frac = max(_layer_counts.values()) / _csize
            if _dominant_frac > 0.9:
                _pure_count += _csize

        _pure_fracs.append(_pure_count / _total)
        _mixed_fracs.append(1.0 - _pure_count / _total)

    # Stacked bar: for each checkpoint, show fraction of params per layer
    # colored by whether they end up in a cluster dominated by their own layer
    _fig = go.Figure()

    for _fd_idx, _fd in enumerate(fiedler_data):
        _Z = _fd["linkage"]
        _cluster_ids = fcluster(_Z, t=_k, criterion="maxclust")

    # Simpler and more informative: stacked bar showing layer composition
    # of each cluster, aggregated as fraction of total params
    _fig = go.Figure()

    _step_labels = [f"{_fd['step']}" for _fd in fiedler_data]

    for _ln in _layer_names:
        _fracs_per_step = []
        for _fd in fiedler_data:
            _Z = _fd["linkage"]
            _cluster_ids = fcluster(_Z, t=_k, criterion="maxclust")
            _total = len(_cluster_ids)
            _layer_mask = _labels_arr == _ln
            _fracs_per_step.append(np.sum(_layer_mask) / _total)

        _fig.add_trace(go.Bar(
            x=_step_labels,
            y=_fracs_per_step,
            name=_ln,
            marker_color=LAYER_COLORS[_ln],
            hovertemplate=f"{_ln}<br>Step %{{x}}<br>Fraction: %{{y:.3f}}<extra></extra>",
        ))

    # Now overlay: fraction of params in "correctly grouped" clusters
    # (clusters where >90% of params come from a single layer)
    _purity_per_step = []
    for _fd in fiedler_data:
        _Z = _fd["linkage"]
        _cluster_ids = fcluster(_Z, t=_k, criterion="maxclust")
        _total = len(_cluster_ids)
        _correct = 0
        for _cid in range(1, _k + 1):
            _cmask = _cluster_ids == _cid
            _csize = np.sum(_cmask)
            if _csize == 0:
                continue
            _layer_counts = [np.sum(_cmask & (_labels_arr == _ln)) for _ln in _layer_names]
            _dominant = max(_layer_counts)
            if _dominant / _csize > 0.9:
                _correct += _dominant
        _purity_per_step.append(_correct / _total)

    _fig.add_trace(go.Scatter(
        x=_step_labels,
        y=_purity_per_step,
        mode="lines+markers",
        name="Cluster purity (>90%)",
        line=dict(color="#f9a8d4", width=2.5),
        marker=dict(size=6, symbol="diamond"),
        yaxis="y2",
        hovertemplate="Step %{x}<br>Purity: %{y:.1%}<extra></extra>",
    ))

    _fig.update_layout(
        title=f"Layer Composition & Cluster Purity (k={_k})",
        xaxis_title="Training step",
        yaxis_title="Fraction of parameters",
        yaxis2=dict(
            title="Cluster purity",
            overlaying="y",
            side="right",
            range=[0, 1.05],
        ),
        barmode="stack",
        height=500,
        legend=dict(font=dict(size=11)),
    )

    composition_fig = mo.ui.plotly(_fig)
    composition_fig
    return (composition_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Side-by-Side: Eigenvalue vs Fiedler Dendrogram

        **Left:** Eigenvalue-spectrum dendrogram (single-linkage clustering of
        sorted eigenvalues by spectral gaps).
        **Right:** Fiedler parameter dendrogram (recursive bipartition of parameters
        by Hessian coupling).

        These show fundamentally different things: the eigenvalue dendrogram captures
        *spectral structure* (how eigenvalues cluster at different resolutions), while
        the Fiedler dendrogram captures *parameter coupling structure* (which parameters
        interact most strongly through second-order effects).
        """
    )
    return


@app.cell(hide_code=True)
def _(
    fiedler_data, fiedler_step_slider, mo, np, plt,
    scipy_dendrogram, spectra,
):
    from scipy.cluster.hierarchy import linkage as scipy_linkage

    _idx = fiedler_step_slider.value
    _fd = fiedler_data[_idx]
    _sp = spectra[_idx]

    # Left: eigenvalue dendrogram
    _evals = _sp["eigenvalues"]
    _Z_eig = scipy_linkage(_evals.reshape(-1, 1), method="single")

    _fig_left, _ax_left = plt.subplots(figsize=(7, 5), facecolor="#111111")
    _ax_left.set_facecolor("#111111")
    scipy_dendrogram(
        _Z_eig, ax=_ax_left, truncate_mode="lastp", p=30,
        show_leaf_counts=True, no_labels=False,
        color_threshold=0, above_threshold_color="#fcd34d",
        leaf_font_size=8,
    )
    _ax_left.set_ylabel("Gap size", fontsize=11, color="#e2e8f0")
    _ax_left.set_title(
        f"Eigenvalue Dendrogram — Step {_fd['step']}",
        fontsize=12, color="#e2e8f0",
    )
    _ax_left.tick_params(colors="#94a3b8")
    _ax_left.spines["bottom"].set_color("#334155")
    _ax_left.spines["left"].set_color("#334155")
    _ax_left.spines["top"].set_visible(False)
    _ax_left.spines["right"].set_visible(False)
    _fig_left.tight_layout()

    # Right: Fiedler dendrogram
    _Z_fied = _fd["linkage"]

    _fig_right, _ax_right = plt.subplots(figsize=(7, 5), facecolor="#111111")
    _ax_right.set_facecolor("#111111")
    scipy_dendrogram(
        _Z_fied, ax=_ax_right, truncate_mode="lastp", p=30,
        show_leaf_counts=True, no_labels=False,
        color_threshold=0, above_threshold_color="#7dd3fc",
        leaf_font_size=8,
    )
    _ax_right.set_ylabel("Merge height (1/lambda_2)", fontsize=11, color="#e2e8f0")
    _ax_right.set_title(
        f"Fiedler Dendrogram — Step {_fd['step']}",
        fontsize=12, color="#e2e8f0",
    )
    _ax_right.tick_params(colors="#94a3b8")
    _ax_right.spines["bottom"].set_color("#334155")
    _ax_right.spines["left"].set_color("#334155")
    _ax_right.spines["top"].set_visible(False)
    _ax_right.spines["right"].set_visible(False)
    _fig_right.tight_layout()

    _comparison = mo.hstack([
        _fig_left,
        _fig_right,
    ], justify="center", gap=1)

    plt.close(_fig_left)
    plt.close(_fig_right)

    _stats = mo.md(
        f"**Step {_fd['step']}** (epoch {_fd['epoch']}) — "
        f"loss={_fd['train_loss']:.4f}, acc={_fd['test_acc']:.1%} "
        f"&nbsp;|&nbsp; Eigenvalues: [{np.min(_evals):.3f}, {np.max(_evals):.3f}] "
        f"&nbsp;|&nbsp; Fiedler root height: {_Z_fied[-1, 2]:.2e}"
    )

    mo.vstack([_comparison, _stats])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        *Fiedler Bipartition Experiment — LeNet-tiny on MNIST, 5,994 parameters,
        full Hessian computed at 25 checkpoints over 30 epochs of SGD training.
        Dendrograms built by recursive spectral bipartition of the Hessian coupling
        graph $|H_{ij}|$ using the Fiedler vector.*
        """
    )
    return


if __name__ == "__main__":
    app.run()
