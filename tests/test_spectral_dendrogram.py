import numpy as np
import pytest
from src.spectral_dendrogram import fiedler_split


def test_fiedler_split_block_diagonal():
    """Block-diagonal Hessian should split cleanly along the blocks."""
    # Two 3x3 blocks with no cross-coupling
    H = np.zeros((6, 6))
    H[:3, :3] = np.array([[2.0, 0.5, 0.3],
                           [0.5, 1.5, 0.4],
                           [0.3, 0.4, 1.0]])
    H[3:, 3:] = np.array([[3.0, 0.8, 0.2],
                           [0.8, 2.0, 0.6],
                           [0.2, 0.6, 1.5]])

    left, right, lambda2 = fiedler_split(H)

    # Should split into {0,1,2} and {3,4,5} (or vice versa)
    assert set(left) | set(right) == {0, 1, 2, 3, 4, 5}
    assert len(set(left) & set(right)) == 0
    assert (set(left) == {0, 1, 2} and set(right) == {3, 4, 5}) or \
           (set(left) == {3, 4, 5} and set(right) == {0, 1, 2})
    assert lambda2 >= 0


def test_fiedler_split_weak_coupling():
    """Blocks with weak cross-coupling should still split correctly."""
    H = np.zeros((6, 6))
    H[:3, :3] = np.array([[2.0, 0.5, 0.3],
                           [0.5, 1.5, 0.4],
                           [0.3, 0.4, 1.0]])
    H[3:, 3:] = np.array([[3.0, 0.8, 0.2],
                           [0.8, 2.0, 0.6],
                           [0.2, 0.6, 1.5]])
    # Add weak cross-block coupling
    H[0, 3] = H[3, 0] = 0.01
    H[1, 4] = H[4, 1] = 0.02

    left, right, lambda2 = fiedler_split(H)

    assert (set(left) == {0, 1, 2} and set(right) == {3, 4, 5}) or \
           (set(left) == {3, 4, 5} and set(right) == {0, 1, 2})
    assert lambda2 > 0


def test_fiedler_split_two_elements():
    """Minimal split: 2 elements."""
    H = np.array([[1.0, 0.1],
                   [0.1, 2.0]])
    left, right, lambda2 = fiedler_split(H)
    assert len(left) == 1 and len(right) == 1
    assert set(left) | set(right) == {0, 1}
