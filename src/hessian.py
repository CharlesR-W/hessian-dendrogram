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


def _make_loss_fn(
    model: torch.nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[callable, torch.Tensor]:
    """Create a loss function mapping flat params -> scalar, for Hessian computation.

    Returns (loss_fn, flat_params) where flat_params is detached float64.
    """
    model.eval()
    param_shapes = {name: p.shape for name, p in model.named_parameters()}
    param_names = list(param_shapes.keys())

    flat_params = torch.nn.utils.parameters_to_vector(model.parameters())
    flat_params = flat_params.detach().double().requires_grad_(True)

    data_f64 = data.double()

    def loss_fn(flat_p):
        param_dict = {}
        offset = 0
        for name in param_names:
            shape = param_shapes[name]
            numel = 1
            for s in shape:
                numel *= s
            param_dict[name] = flat_p[offset:offset + numel].view(shape)
            offset += numel
        out = torch.func.functional_call(model, param_dict, (data_f64,))
        return torch.nn.functional.cross_entropy(out, targets)

    return loss_fn, flat_params


def compute_model_hessian(
    model: torch.nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    hvp_batch_size: int = 64,
) -> torch.Tensor:
    """Compute full Hessian of cross-entropy loss for a model on given data.

    Uses batched Hessian-vector products via vmap(jvp(grad)). This avoids
    the OOM of torch.func.hessian (which vmaps over all N directions at once)
    by processing hvp_batch_size directions at a time.

    Args:
        model: Neural network (will be put in eval mode).
        data: Input tensor (N, C, H, W).
        targets: Target labels (N,).
        hvp_batch_size: Number of HVP directions to process in parallel.
            Higher = faster but more memory. 64 is a safe default.

    Returns:
        P x P Hessian matrix where P = number of model parameters.
    """
    loss_fn, flat_params = _make_loss_fn(model, data, targets)
    N = flat_params.shape[0]

    grad_fn = torch.func.grad(loss_fn)

    def hvp_single(v):
        return torch.func.jvp(grad_fn, (flat_params,), (v,))[1]

    batched_hvp = torch.vmap(hvp_single)

    H = torch.zeros(N, N, dtype=torch.float64)
    for start in range(0, N, hvp_batch_size):
        end = min(start + hvp_batch_size, N)
        basis = torch.zeros(end - start, N, dtype=torch.float64)
        for j in range(end - start):
            basis[j, start + j] = 1.0
        H[start:end] = batched_hvp(basis).detach()

    # Symmetrize (numerical errors can make it slightly asymmetric)
    H = (H + H.T) / 2
    return H
