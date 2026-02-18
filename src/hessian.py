"""Hessian eigenspectrum computation: full for small models, Lanczos for large."""

import torch
import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, eigsh


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

    data_f64 = data.double() if data.is_floating_point() else data

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


def compute_lanczos_eigenspectrum(
    model: torch.nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    k: int = 200,
    n_save_vectors: int = 50,
) -> tuple[NDArray, NDArray]:
    """Compute top-k and bottom-k Hessian eigenvalues via Lanczos iteration.

    Uses Hessian-vector products (never forms the full matrix), so this scales
    to models with 100K+ parameters where the full Hessian would be infeasible.

    Returns eigenvalues sorted ascending (bottom-k concatenated with top-k,
    deduplicated), and the corresponding eigenvectors for the extremes.
    """
    loss_fn, flat_params = _make_loss_fn(model, data, targets)
    N = flat_params.shape[0]
    grad_fn = torch.func.grad(loss_fn)

    def hvp(v_np):
        v = torch.from_numpy(v_np).double()
        hv = torch.func.jvp(grad_fn, (flat_params,), (v,))[1]
        return hv.detach().numpy()

    H_op = LinearOperator((N, N), matvec=hvp, dtype=np.float64)

    actual_k = min(k, N // 2 - 1)

    # Largest eigenvalues
    evals_top, evecs_top = eigsh(H_op, k=actual_k, which='LA', tol=1e-6)
    # Smallest (most negative) eigenvalues
    evals_bot, evecs_bot = eigsh(H_op, k=actual_k, which='SA', tol=1e-6)

    # Combine and deduplicate
    all_evals = np.concatenate([evals_bot, evals_top])
    all_evecs = np.concatenate([evecs_bot, evecs_top], axis=1)
    order = np.argsort(all_evals)
    all_evals = all_evals[order]
    all_evecs = all_evecs[:, order]

    # Deduplicate close eigenvalues (Lanczos may find overlapping sets)
    mask = np.ones(len(all_evals), dtype=bool)
    for i in range(1, len(all_evals)):
        if abs(all_evals[i] - all_evals[i - 1]) < 1e-8 * (abs(all_evals[i]) + 1):
            mask[i] = False
    all_evals = all_evals[mask]
    all_evecs = all_evecs[:, mask]

    # Save extreme eigenvectors
    sv = min(n_save_vectors, len(all_evals) // 2)
    if 2 * sv >= len(all_evals):
        saved_evecs = all_evecs
    else:
        saved_evecs = np.concatenate([all_evecs[:, :sv], all_evecs[:, -sv:]], axis=1)

    return all_evals, saved_evecs
