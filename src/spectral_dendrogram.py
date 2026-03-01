"""Recursive spectral bipartition of the Hessian coupling graph.

Builds a dendrogram over parameters (not eigenvalues) using the Fiedler vector
of the graph Laplacian constructed from |H_ij|.  Two parameters in the same
cluster have strong second-order interaction in the loss landscape.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh


def fiedler_split(H: NDArray) -> tuple[NDArray, NDArray, float]:
    """Single spectral bipartition of parameters via the Fiedler vector.

    Constructs a coupling graph from |H_ij| (zeroed diagonal), computes the
    graph Laplacian, and splits parameters by the median of the Fiedler vector.

    Args:
        H: Symmetric (N, N) Hessian matrix.

    Returns:
        (left_indices, right_indices, lambda2) where lambda2 is the algebraic
        connectivity (2nd smallest eigenvalue of the Laplacian).
    """
    N = H.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 parameters to split")

    # Affinity: |H_ij|, zero diagonal
    A = np.abs(H)
    np.fill_diagonal(A, 0.0)

    # Graph Laplacian: L = D - A
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Fiedler vector: 2nd smallest eigenvector of L
    eigenvalues, eigenvectors = eigh(L, subset_by_index=[0, 1])
    lambda2 = eigenvalues[1]
    fiedler = eigenvectors[:, 1]

    # Median split (prevents degenerate 1-vs-(N-1) on dense graphs)
    median = np.median(fiedler)
    left = np.where(fiedler <= median)[0]
    right = np.where(fiedler > median)[0]

    # Handle exact median ties: ensure both sides non-empty
    if len(left) == 0:
        left = np.array([0])
        right = np.arange(1, N)
    elif len(right) == 0:
        right = np.array([N - 1])
        left = np.arange(0, N - 1)

    return left, right, float(lambda2)
