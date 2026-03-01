"""Recursive spectral bipartition of the Hessian coupling graph.

Builds a dendrogram over parameters (not eigenvalues) using the Fiedler vector
of the graph Laplacian constructed from |H_ij|.  Two parameters in the same
cluster have strong second-order interaction in the loss landscape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch.nn as nn
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


# ---------------------------------------------------------------------------
# Recursive dendrogram construction
# ---------------------------------------------------------------------------

@dataclass
class _TreeNode:
    """Binary tree node for recursive bipartition."""
    indices: NDArray          # original parameter indices in this cluster
    left: Optional[_TreeNode] = None
    right: Optional[_TreeNode] = None
    height: float = 0.0      # merge height (1/lambda2)


def _recursive_bipartition(
    H: NDArray,
    indices: NDArray,
    max_depth: int = 50,
    _depth: int = 0,
) -> _TreeNode:
    """Recursively bipartition parameters using the Fiedler vector.

    Args:
        H: Full (N, N) Hessian matrix.
        indices: Current subset of parameter indices (into the full H).
        max_depth: Safety limit on recursion depth.
        _depth: Current recursion depth (internal).

    Returns:
        _TreeNode representing the recursive partition tree.
    """
    n = len(indices)

    # Base case: leaf
    if n <= 1:
        return _TreeNode(indices=indices, height=0.0)

    # Base case: pair - just merge directly
    if n == 2:
        sub_H = H[np.ix_(indices, indices)]
        # Use |H_01| as coupling strength; height = 1/coupling (or large if zero)
        coupling = abs(sub_H[0, 1])
        height = 1.0 / coupling if coupling > 1e-15 else 1e6
        left_child = _TreeNode(indices=indices[:1], height=0.0)
        right_child = _TreeNode(indices=indices[1:], height=0.0)
        return _TreeNode(indices=indices, left=left_child, right=right_child,
                         height=height)

    # Depth guard
    if _depth >= max_depth:
        return _TreeNode(indices=indices, height=0.0)

    # Extract sub-Hessian and split
    sub_H = H[np.ix_(indices, indices)]
    left_local, right_local, lambda2 = fiedler_split(sub_H)

    # Merge height = 1/lambda2 (inverse algebraic connectivity)
    height = 1.0 / lambda2 if lambda2 > 1e-15 else 1e6

    # Map local indices back to original indices
    left_indices = indices[left_local]
    right_indices = indices[right_local]

    # Recurse
    left_child = _recursive_bipartition(H, left_indices, max_depth, _depth + 1)
    right_child = _recursive_bipartition(H, right_indices, max_depth, _depth + 1)

    # Enforce monotonicity: parent height >= max(child heights) + epsilon
    min_height = max(left_child.height, right_child.height) + 1e-10
    height = max(height, min_height)

    return _TreeNode(indices=indices, left=left_child, right=right_child,
                     height=height)


def _tree_to_linkage(root: _TreeNode, n: int) -> NDArray:
    """Convert a binary tree to scipy linkage matrix format.

    Args:
        root: Root of the binary partition tree.
        n: Total number of original leaf nodes.

    Returns:
        (n-1, 4) linkage matrix: [left_id, right_id, height, count]
    """
    merges: list[list[float]] = []
    next_id = n  # internal node IDs start at n

    def _walk(node: _TreeNode) -> int:
        """Post-order walk; returns the linkage node ID for this node."""
        nonlocal next_id

        # Leaf: single original index
        if node.left is None and node.right is None:
            if len(node.indices) == 1:
                return int(node.indices[0])
            else:
                # Multi-element leaf (from depth cutoff) - chain merges
                return _chain_merge(node)

        left_id = _walk(node.left)
        right_id = _walk(node.right)

        merge_id = next_id
        next_id += 1
        count = (
            _count(node.left) + _count(node.right)
        )
        merges.append([float(left_id), float(right_id), node.height, float(count)])
        return merge_id

    def _count(node: _TreeNode) -> int:
        """Count leaves under a node."""
        if node.left is None and node.right is None:
            return len(node.indices)
        return _count(node.left) + _count(node.right)

    def _chain_merge(node: _TreeNode) -> int:
        """Chain-merge a multi-element leaf into sequential binary merges."""
        nonlocal next_id
        idx_list = list(node.indices)
        current_id = int(idx_list[0])
        for i in range(1, len(idx_list)):
            merge_id = next_id
            next_id += 1
            merges.append([
                float(current_id),
                float(idx_list[i]),
                node.height,
                float(i + 1),
            ])
            current_id = merge_id
        return current_id

    _walk(root)

    return np.array(merges, dtype=np.float64)


def build_fiedler_dendrogram(
    H: NDArray,
    max_depth: int = 50,
) -> NDArray:
    """Build a full dendrogram via recursive spectral bipartition.

    Args:
        H: Symmetric (N, N) Hessian matrix.
        max_depth: Maximum recursion depth (default 50).

    Returns:
        Scipy-compatible (N-1, 4) linkage matrix Z where each row is
        [left_id, right_id, merge_height, cluster_size].
    """
    n = H.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 parameters for a dendrogram")

    indices = np.arange(n)
    root = _recursive_bipartition(H, indices, max_depth=max_depth)
    Z = _tree_to_linkage(root, n)

    # Final monotonicity pass: ensure Z[:, 2] is non-decreasing
    for i in range(1, len(Z)):
        if Z[i, 2] < Z[i - 1, 2]:
            Z[i, 2] = Z[i - 1, 2] + 1e-10

    return Z


# ---------------------------------------------------------------------------
# Parameter-to-layer label mapping
# ---------------------------------------------------------------------------

def parameter_layer_labels(model: nn.Module) -> list[str]:
    """Map each flat parameter index to its layer/module name.

    For a parameter named 'features.0.weight', the label is 'features.0'.
    For 'classifier.bias', the label is 'classifier'.

    Args:
        model: A PyTorch nn.Module.

    Returns:
        List of length sum(p.numel() for p in model.parameters()), where
        each element is the module name owning that parameter.
    """
    labels: list[str] = []
    for name, param in model.named_parameters():
        # Strip .weight / .bias suffix to get module name
        parts = name.rsplit(".", 1)
        module_name = parts[0] if len(parts) > 1 else name
        labels.extend([module_name] * param.numel())
    return labels
