"""Dendrogram construction from Hessian eigenvalue spectra.

The core operation: single-linkage hierarchical clustering on eigenvalues,
which implements the epsilon-coarse-graining construction from the research
notes.  For 1D data, this is equivalent to merging eigenvalues separated
by gaps smaller than epsilon.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import linkage, fcluster


def build_dendrogram(eigenvalues: NDArray) -> NDArray:
    """Build single-linkage dendrogram from eigenvalue spectrum.

    Args:
        eigenvalues: 1D array of eigenvalues (any order, may include negatives).

    Returns:
        Linkage matrix Z (scipy format): (N-1, 4) array.
        Returns empty (0, 4) array if fewer than 2 eigenvalues.
    """
    if len(eigenvalues) < 2:
        return np.empty((0, 4))
    return linkage(eigenvalues.reshape(-1, 1), method="single")


def cluster_count_at_epsilon(Z: NDArray, epsilon: float, n: int) -> int:
    """Number of clusters at coarse-graining resolution epsilon.

    Args:
        Z: Linkage matrix from build_dendrogram.
        epsilon: Resolution parameter (merge threshold).
        n: Number of original data points.

    Returns:
        Number of clusters.
    """
    if n <= 1 or Z.shape[0] == 0:
        return n
    labels = fcluster(Z, t=epsilon, criterion="distance")
    return len(set(labels))


def cluster_count_curve(
    Z: NDArray,
    n: int,
    epsilon_range: NDArray | None = None,
    n_points: int = 200,
) -> tuple[NDArray, NDArray]:
    """Compute n(epsilon) -- number of clusters as a function of resolution.

    Args:
        Z: Linkage matrix.
        n: Number of original data points.
        epsilon_range: Array of epsilon values to evaluate.  If None, auto-range
            from min merge height to max merge height (log-spaced).
        n_points: Number of epsilon points if auto-ranging.

    Returns:
        (epsilons, counts) arrays.
    """
    if Z.shape[0] == 0:
        return np.array([1.0]), np.array([n])

    merge_heights = Z[:, 2]
    lo = merge_heights.min() * 0.5
    hi = merge_heights.max() * 2.0
    if lo <= 0:
        lo = 1e-12

    if epsilon_range is None:
        epsilon_range = np.logspace(np.log10(lo), np.log10(hi), n_points)

    counts = np.array([cluster_count_at_epsilon(Z, eps, n) for eps in epsilon_range])
    return epsilon_range, counts


def extract_top_gaps(eigenvalues: NDArray, k: int = 10) -> NDArray:
    """Extract the k largest gaps between adjacent sorted eigenvalues.

    Args:
        eigenvalues: 1D array of eigenvalues.
        k: Number of top gaps to return.

    Returns:
        Array of top-k gap sizes, sorted descending.
    """
    sorted_evals = np.sort(eigenvalues)
    gaps = np.diff(sorted_evals)
    top_k = np.sort(gaps)[::-1][:k]
    return top_k
