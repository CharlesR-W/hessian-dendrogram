import torch
import numpy as np
from src.hessian import compute_hessian, compute_eigenspectrum, compute_model_hessian


def test_hessian_quadratic():
    """For f(x) = 0.5 * x^T A x, the Hessian is exactly A."""
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    x0 = torch.tensor([1.0, 2.0], dtype=torch.float64)

    def f(x):
        return 0.5 * x @ A @ x

    H = compute_hessian(f, x0)
    np.testing.assert_allclose(H.numpy(), A.numpy(), atol=1e-10)


def test_hessian_symmetric():
    """Hessian must be symmetric."""
    A = torch.tensor([[4.0, 1.5, 0.5],
                      [1.5, 3.0, 0.8],
                      [0.5, 0.8, 2.0]], dtype=torch.float64)
    x0 = torch.randn(3, dtype=torch.float64)

    def f(x):
        return 0.5 * x @ A @ x + 0.1 * (x ** 3).sum()

    H = compute_hessian(f, x0)
    np.testing.assert_allclose(H.numpy(), H.T.numpy(), atol=1e-10)


def test_eigenspectrum_known():
    """Eigenvalues of a known matrix."""
    A = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    eigenvalues, eigenvectors = compute_eigenspectrum(A)
    expected = np.array([1.0, 3.0])
    np.testing.assert_allclose(eigenvalues, expected, atol=1e-10)
    assert eigenvectors.shape == (2, 2)


def test_eigenspectrum_sorted():
    """Eigenvalues must be returned sorted ascending."""
    A = torch.randn(10, 10, dtype=torch.float64)
    A = A + A.T  # make symmetric
    eigenvalues, _ = compute_eigenspectrum(A)
    assert np.all(np.diff(eigenvalues) >= -1e-12), "Eigenvalues not sorted"


def test_model_hessian_shape():
    """Hessian of LeNet-tiny has the right shape and is symmetric."""
    from src.model import LeNetTiny

    model = LeNetTiny()
    n_params = sum(p.numel() for p in model.parameters())

    # Tiny batch for speed
    data = torch.randn(8, 1, 28, 28)
    targets = torch.randint(0, 10, (8,))

    H = compute_model_hessian(model, data, targets)
    assert H.shape == (n_params, n_params), f"Expected ({n_params}, {n_params}), got {H.shape}"

    # Symmetry
    np.testing.assert_allclose(H.numpy(), H.T.numpy(), atol=1e-8)
