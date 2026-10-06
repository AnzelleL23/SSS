"""
Simple CPU and GPU implementations of Semi-Stochastic Sinkhorn (SSS).

The CPU solver uses single-coordinate updates.
The GPU solver uses minibatch/block updates without materializing the full
m x n Gibbs kernel.

Both implementations use the canonical maximal SSS step:
    gamma = 1 / n          (single-coordinate CPU),
    gamma = |S| / n        (block GPU),
so the scaling update reduces to an exact correction of the selected
target coordinate/block.

The implementations assume the second marginal is the larger one (m <= n).
If m > n, swap the two marginals/supports (and transpose the cost matrix
for the CPU solver).

Scaling convention
------------------
u = mu * exp(f)
v = nu * exp(g)

The transport plan is
    pi = diag(u) K diag(v),
where K = exp(-C / reg).
"""

import math
import numpy as np

try:
    import torch
except ImportError:
    torch = None


# ---------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------

def _normalize_numpy_marginal(p, name):
    p = np.asarray(p, dtype=np.float64)

    if p.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if np.any(p <= 0):
        raise ValueError(f"{name} must have strictly positive entries.")

    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"{name} must have positive finite mass.")

    return p / total


def marginal_error_cpu(a, b, K, u, v, Kv=None):
    """Return ||pi 1 - a||_1 + ||pi^T 1 - b||_1."""
    if Kv is None:
        Kv = K @ v

    row_marginal = u * Kv
    col_marginal = v * (K.T @ u)

    return float(
        np.sum(np.abs(row_marginal - a))
        + np.sum(np.abs(col_marginal - b))
    )


def sss_cpu(a, b, C, reg, tol=1e-6, max_epochs=10_000,
            sampling="random", seed=None):
    """
    Solve discrete EOT using single-coordinate SSS on CPU.

    Parameters
    ----------
    a : array-like, shape (m,)
        Source marginal.
    b : array-like, shape (n,)
        Target marginal. This should be the larger marginal (m <= n).
    C : array-like, shape (m, n)
        Cost matrix.
    reg : float
        Entropic regularization parameter.
    tol : float, default=1e-6
        Stop when the total L1 marginal error is <= tol.
    max_epochs : int, default=10000
        Safety limit. One epoch consists of n coordinate updates.
    sampling : {"random", "shuffle", "cyclic"}, default="random"
        random  : n i.i.d. coordinates with replacement per epoch.
        shuffle : a fresh random permutation each epoch.
        cyclic  : one random permutation sampled once and reused.
    seed : int or None
        Random seed.

    Returns
    -------
    u : ndarray, shape (m,)
    v : ndarray, shape (n,)
        Scaling variables such that pi = diag(u) K diag(v).
    info : dict
        Contains final error, number of epochs/updates, and convergence flag.
    """
    if reg <= 0:
        raise ValueError("reg must be positive.")
    if tol < 0:
        raise ValueError("tol must be nonnegative.")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")
    if sampling not in {"random", "shuffle", "cyclic"}:
        raise ValueError("sampling must be 'random', 'shuffle', or 'cyclic'.")

    a = _normalize_numpy_marginal(a, "a")
    b = _normalize_numpy_marginal(b, "b")
    C = np.asarray(C, dtype=np.float64)

    if C.ndim != 2:
        raise ValueError("C must be a two-dimensional cost matrix.")

    m, n = C.shape
    if len(a) != m or len(b) != n:
        raise ValueError("C must have shape (len(a), len(b)).")
    if m > n:
        raise ValueError(
            "This SSS implementation updates the second (larger) marginal. "
            "For m > n, call sss_cpu(b, a, C.T, ...) and swap the returned "
            "scalings."
        )

    K = np.exp(-C / reg)
    tiny = np.finfo(np.float64).tiny
    rng = np.random.default_rng(seed)

    # g^1 = 0  <=>  v^1 = b
    v = b.copy()
    Kv = K @ v
    u = a / np.maximum(Kv, tiny)

    error = marginal_error_cpu(a, b, K, u, v, Kv)
    epochs = 0
    updates = 0

    fixed_order = rng.permutation(n) if sampling == "cyclic" else None

    while error > tol and epochs < max_epochs:
        if sampling == "shuffle":
            order = rng.permutation(n)
        elif sampling == "cyclic":
            order = fixed_order
        else:
            order = None

        for t in range(n):
            j = int(rng.integers(n)) if sampling == "random" else int(order[t])

            # Selected target marginal:
            # (Y_# pi)_j = v_j * K_:j^T u.
            denom = max(float(np.dot(K[:, j], u)), tiny)

            # Canonical gamma = 1/n makes the SSS update
            # v_j <- b_j / (K_:j^T u).
            old_v_j = v[j]
            new_v_j = b[j] / denom
            v[j] = new_v_j

            # Rolling update of K v; avoids recomputing K @ v from scratch.
            Kv += K[:, j] * (new_v_j - old_v_j)

            # Restore the source marginal exactly (up to floating point).
            u = a / np.maximum(Kv, tiny)

            updates += 1

        epochs += 1
        error = marginal_error_cpu(a, b, K, u, v, Kv)

    info = {
        "error": error,
        "epochs": epochs,
        "updates": updates,
        "converged": error <= tol,
    }

    return u, v, info


def transport_plan_cpu(u, v, C, reg):
    """Materialize pi = diag(u) exp(-C/reg) diag(v)."""
    C = np.asarray(C, dtype=np.float64)
    K = np.exp(-C / reg)
    return u[:, None] * K * v[None, :]


# ---------------------------------------------------------------------
# GPU / PyTorch minibatch implementation
# ---------------------------------------------------------------------

def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required for sss_gpu.")


def _normalize_torch_marginal(p, length, device, dtype, name):
    if p is None:
        return torch.full(
            (length,), 1.0 / length, device=device, dtype=dtype
        )

    p = torch.as_tensor(p, device=device, dtype=dtype)

    if p.ndim != 1 or p.numel() != length:
        raise ValueError(f"{name} must have shape ({length},).")
    if torch.any(p <= 0):
        raise ValueError(f"{name} must have strictly positive entries.")

    total = p.sum()
    if not torch.isfinite(total) or total.item() <= 0:
        raise ValueError(f"{name} must have positive finite mass.")

    return p / total


@torch.no_grad() if torch is not None else (lambda f: f)
def kernel_block_squared_euclidean(X, Y_block, reg):
    """
    Compute exp(-||x-y||^2/reg) for one X-by-Y_block kernel block.

    The full m x n kernel is never materialized.
    """
    X2 = torch.sum(X * X, dim=1, keepdim=True)
    Y2 = torch.sum(Y_block * Y_block, dim=1, keepdim=True).T

    C_block = X2 + Y2 - 2.0 * (X @ Y_block.T)
    C_block.clamp_min_(0.0)
    C_block.mul_(-1.0 / reg)
    C_block.exp_()

    return C_block


@torch.no_grad() if torch is not None else (lambda f: f)
def marginal_error_gpu(X, Y, a, b, u, v, reg, batch_size):
    """
    Compute the total L1 marginal error without storing the full kernel.
    """
    m, n = X.shape[0], Y.shape[0]

    row_sum = torch.zeros(m, device=X.device, dtype=X.dtype)
    col_error = torch.zeros((), device=X.device, dtype=X.dtype)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        K = kernel_block_squared_euclidean(X, Y[start:end], reg)
        v_block = v[start:end]

        row_sum += K @ v_block
        col_marginal = v_block * (K.T @ u)
        col_error += torch.sum(torch.abs(col_marginal - b[start:end]))

    row_marginal = u * row_sum
    row_error = torch.sum(torch.abs(row_marginal - a))

    return float((row_error + col_error).item())


@torch.no_grad() if torch is not None else (lambda f: f)
def sss_gpu(X, Y, reg, batch_size=8192, a=None, b=None, tol=1e-6,
            max_epochs=1000, sampling="random", seed=1337):
    """
    Solve discrete EOT using minibatch SSS with PyTorch.

    This implementation is designed for large problems: it constructs only
    an m x batch_size Gibbs-kernel block at a time.

    Parameters
    ----------
    X : torch.Tensor, shape (m, d)
        Source support. Put X on the GPU before calling this function.
    Y : torch.Tensor, shape (n, d)
        Target support, on the same device/dtype as X.
        This should be the larger support (m <= n).
    reg : float
        Entropic regularization parameter.
    batch_size : int, default=8192
        Number of target coordinates updated in one block.
    a : array-like or torch.Tensor, shape (m,), optional
        Source marginal. Uniform if omitted.
    b : array-like or torch.Tensor, shape (n,), optional
        Target marginal. Uniform if omitted.
    tol : float, default=1e-6
        Stop when the total L1 marginal error is <= tol.
    max_epochs : int, default=1000
        Safety limit. One epoch consists of ceil(n/batch_size) block updates.
    sampling : {"random", "shuffle", "cyclic"}, default="random"
        random  : blocks sampled with replacement.
        shuffle : fresh random permutation of blocks each epoch.
        cyclic  : one random block permutation sampled once and reused.
    seed : int, default=1337
        Random seed controlling block order.

    Returns
    -------
    u : torch.Tensor, shape (m,)
    v : torch.Tensor, shape (n,)
        Scaling variables such that pi = diag(u) K diag(v).
    info : dict
        Contains final error, number of epochs/block updates, and convergence
        flag.

    Notes
    -----
    The default kernel is K_ij = exp(-||X_i-Y_j||^2 / reg).
    The maximal block step gamma = |S|/n gives
        v(S) <- b(S) / (K_:S^T u).
    """
    _require_torch()

    if not torch.is_tensor(X) or not torch.is_tensor(Y):
        raise TypeError("X and Y must be torch tensors.")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be two-dimensional tensors.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same feature dimension.")
    if X.device != Y.device:
        raise ValueError("X and Y must be on the same device.")
    if X.dtype != Y.dtype:
        raise ValueError("X and Y must have the same dtype.")
    if not X.dtype.is_floating_point:
        raise TypeError("X and Y must have floating-point dtype.")
    if reg <= 0:
        raise ValueError("reg must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if tol < 0:
        raise ValueError("tol must be nonnegative.")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")
    if sampling not in {"random", "shuffle", "cyclic"}:
        raise ValueError("sampling must be 'random', 'shuffle', or 'cyclic'.")

    m, n = X.shape[0], Y.shape[0]

    if m > n:
        raise ValueError(
            "This SSS implementation updates the second (larger) support. "
            "For m > n, swap X and Y (and a and b), then swap the returned "
            "scalings."
        )

    a = _normalize_torch_marginal(a, m, X.device, X.dtype, "a")
    b = _normalize_torch_marginal(b, n, X.device, X.dtype, "b")

    batch_size = min(batch_size, n)
    num_blocks = math.ceil(n / batch_size)
    tiny = torch.finfo(X.dtype).tiny

    # g^1 = 0  <=>  v^1 = b
    v = b.clone()

    # Initialize Kv without storing the full kernel.
    Kv = torch.zeros(m, device=X.device, dtype=X.dtype)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        K = kernel_block_squared_euclidean(X, Y[start:end], reg)
        Kv += K @ v[start:end]

    u = a / Kv.clamp_min(tiny)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    random_rng = np.random.default_rng(seed)

    fixed_order = None
    if sampling == "cyclic":
        fixed_order = torch.randperm(
            num_blocks, generator=generator
        ).tolist()

    error = marginal_error_gpu(
        X, Y, a, b, u, v, reg, batch_size
    )

    epochs = 0
    block_updates = 0

    updates = 0

    while error > tol and epochs < max_epochs:

        # ---------------------------------------------------------
        # RANDOM: single coordinates sampled with replacement
        # ---------------------------------------------------------
        if sampling == "random":

            for block_id in range(num_blocks):

                q = min(
                    batch_size,
                    n - block_id * batch_size
                )

                idx_np = random_rng.choice(
                    n,
                    size=q,
                    replace=False,
                    shuffle=False
                )

                idx = torch.as_tensor(
                    idx_np,
                    device=Y.device,
                    dtype=torch.long
                )

                K = kernel_block_squared_euclidean(
                    X,
                    Y[idx],
                    reg
                )

                denom = (
                    K.T @ u
                ).clamp_min(tiny)

                old_v = v[idx].clone()

                # b is the target marginal vector here
                new_v = b[idx] / denom

                delta_v = new_v - old_v

                Kv += K @ delta_v
                v[idx] = new_v

                u = a / Kv.clamp_min(tiny)

                updates += 1

        # ---------------------------------------------------------
        # SHUFFLE / CYCLIC
        # ---------------------------------------------------------
        else:
            if sampling == "shuffle":
                block_order = torch.randperm(
                    num_blocks, generator=generator
                ).tolist()
            else:
                block_order = fixed_order

            for block_id in block_order:
                start = block_id * batch_size
                end = min(start + batch_size, n)

                K = kernel_block_squared_euclidean(
                    X, Y[start:end], reg
                )

                denom = (K.T @ u).clamp_min(tiny)

                old_v = v[start:end].clone()

                new_v = b[start:end] / denom
                delta_v = new_v - old_v

                Kv += K @ delta_v
                v[start:end] = new_v

                u = a / Kv.clamp_min(tiny)

                updates += 1

        epochs += 1
        error = marginal_error_gpu(
            X, Y, a, b, u, v, reg, batch_size
        )

    info = {
        "error": error,
        "epochs": epochs,
        "updates": updates,
        "converged": error <= tol,
        }

    return u, v, info
