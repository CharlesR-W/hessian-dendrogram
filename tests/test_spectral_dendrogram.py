import numpy as np
from src.spectral_dendrogram import fiedler_split, build_fiedler_dendrogram


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


def test_build_fiedler_dendrogram_block_diagonal():
    """Full dendrogram from block-diagonal Hessian."""
    H = np.zeros((6, 6))
    H[:3, :3] = np.array([[2.0, 0.5, 0.3],
                           [0.5, 1.5, 0.4],
                           [0.3, 0.4, 1.0]])
    H[3:, 3:] = np.array([[3.0, 0.8, 0.2],
                           [0.8, 2.0, 0.6],
                           [0.2, 0.6, 1.5]])

    Z = build_fiedler_dendrogram(H)

    # Scipy linkage matrix: (N-1, 4) shape
    assert Z.shape == (5, 4)
    # Column 3 is cluster size; final merge should have all 6
    assert Z[-1, 3] == 6
    # Merge heights should be non-decreasing (valid linkage)
    assert np.all(np.diff(Z[:, 2]) >= -1e-12), "Merge heights not monotonic"


def test_build_fiedler_dendrogram_small():
    """Two-element Hessian produces one merge."""
    H = np.array([[1.0, 0.3],
                   [0.3, 2.0]])
    Z = build_fiedler_dendrogram(H)
    assert Z.shape == (1, 4)
    assert Z[0, 3] == 2  # merged cluster has 2 elements


def test_build_fiedler_dendrogram_identity():
    """Identity Hessian: all parameters equally coupled."""
    H = np.eye(4) + 0.1 * np.ones((4, 4))
    Z = build_fiedler_dendrogram(H)
    assert Z.shape == (3, 4)
    assert Z[-1, 3] == 4
    assert np.all(np.diff(Z[:, 2]) >= -1e-12)


def test_parameter_layer_labels():
    """Labels should map each flat param index to its layer name."""
    from src.spectral_dendrogram import parameter_layer_labels
    from src.model import LeNetTiny

    model = LeNetTiny()
    labels = parameter_layer_labels(model)
    n_params = sum(p.numel() for p in model.parameters())

    assert len(labels) == n_params
    # First 200 params are conv1 weight (8*1*5*5=200)
    assert labels[0] == "features.0"
    assert labels[199] == "features.0"
    # Next 8 are conv1 bias
    assert labels[200] == "features.0"
    assert labels[207] == "features.0"
    # Conv2 starts at 208
    assert labels[208] == "features.3"
