"""Full Hessian computation and eigendecomposition for small neural networks."""

import torch
import numpy as np
from numpy.typing import NDArray


def compute_hessian(fn: callable, params: torch.Tensor) -> torch.Tensor:
    """Compute the full Hessian matrix of fn with respect to params.

    Args:
        fn: Scalar-valued function of params.
        params: 1D tensor of parameters.

    Returns:
        N x N Hessian matrix (float64, CPU).
    """
    params = params.detach().double().requires_grad_(True)
    H = torch.func.hessian(fn)(params)
    return H.detach().cpu().double()


def compute_eigenspectrum(
    H: torch.Tensor,
    n_save_vectors: int = 50,
) -> tuple[NDArray, NDArray]:
    """Eigendecompose a symmetric matrix.

    Args:
        H: Symmetric N x N matrix.
        n_save_vectors: Number of top and bottom eigenvectors to return.

    Returns:
        (eigenvalues, eigenvectors) where eigenvalues is sorted ascending 1D array
        and eigenvectors has shape (N, 2*n_save_vectors) -- the n_save_vectors
        smallest and n_save_vectors largest eigenvectors concatenated.
        If N <= 2*n_save_vectors, returns all eigenvectors.
    """
    H_f64 = H.double()
    eigenvalues, eigenvectors = torch.linalg.eigh(H_f64)
    evals = eigenvalues.numpy()
    evecs = eigenvectors.numpy()

    # Save top and bottom eigenvectors
    k = min(n_save_vectors, len(evals))
    if 2 * k >= len(evals):
        saved_evecs = evecs
    else:
        saved_evecs = np.concatenate([evecs[:, :k], evecs[:, -k:]], axis=1)

    return evals, saved_evecs


def compute_model_hessian(
    model: torch.nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute full Hessian of cross-entropy loss for a model on given data.

    Args:
        model: Neural network (will be put in eval mode).
        data: Input tensor (N, C, H, W).
        targets: Target labels (N,).

    Returns:
        P x P Hessian matrix where P = number of model parameters.
    """
    model.eval()

    # Snapshot parameter shapes and names for reconstruction
    param_shapes = {name: p.shape for name, p in model.named_parameters()}
    param_names = list(param_shapes.keys())

    # Flatten current parameters
    flat_params = torch.nn.utils.parameters_to_vector(model.parameters())
    flat_params = flat_params.detach().double().requires_grad_(True)

    def loss_fn(flat_p):
        # Reconstruct parameter dict from flat vector
        param_dict = {}
        offset = 0
        for name in param_names:
            shape = param_shapes[name]
            numel = 1
            for s in shape:
                numel *= s
            param_dict[name] = flat_p[offset:offset + numel].view(shape)
            offset += numel

        # Run model with these parameters
        out = torch.func.functional_call(model, param_dict, (data.double(),))
        return torch.nn.functional.cross_entropy(out, targets)

    H = torch.func.hessian(loss_fn)(flat_params)
    return H.detach().cpu().double()
