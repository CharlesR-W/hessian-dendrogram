# tests/test_dendrogram.py
import numpy as np
from src.dendrogram import (
    build_dendrogram,
    cluster_count_at_epsilon,
    extract_top_gaps,
)


def test_cluster_count_known_spectrum():
    """Known eigenvalues with clear gap structure."""
    # Three clusters: {1.0, 1.1, 1.2}, {5.0, 5.1}, {10.0}
    eigenvalues = np.array([1.0, 1.1, 1.2, 5.0, 5.1, 10.0])

    Z = build_dendrogram(eigenvalues)

    # Below smallest gap: all separate
    assert cluster_count_at_epsilon(Z, 0.05, n=6) == 6
    # Above intra-cluster gaps but below inter-cluster gaps
    assert cluster_count_at_epsilon(Z, 0.15, n=6) == 3
    # Above the 3.8 gap
    assert cluster_count_at_epsilon(Z, 4.0, n=6) == 2
    # Above everything
    assert cluster_count_at_epsilon(Z, 5.0, n=6) == 1


def test_top_gaps():
    eigenvalues = np.array([1.0, 1.1, 1.2, 5.0, 5.1, 10.0])
    gaps = extract_top_gaps(eigenvalues, k=3)
    # Sorted descending: 4.9, 3.8, 0.1
    np.testing.assert_allclose(gaps, [4.9, 3.8, 0.1])


def test_dendrogram_single_element():
    """Edge case: single eigenvalue."""
    eigenvalues = np.array([3.14])
    Z = build_dendrogram(eigenvalues)
    assert Z.shape == (0, 4)  # No merges possible


def test_dendrogram_with_negatives():
    """Negative eigenvalues should work fine."""
    eigenvalues = np.array([-5.0, -4.9, -1.0, 0.0, 1.0, 1.1, 10.0])
    Z = build_dendrogram(eigenvalues)
    # Should have 6 merges (N-1)
    assert Z.shape == (6, 4)
    # At large epsilon, one cluster
    assert cluster_count_at_epsilon(Z, 20.0, n=7) == 1
