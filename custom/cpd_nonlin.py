import numpy as np


def calc_scatters(K: np.ndarray) -> np.ndarray:
    """
    Calculate scatter matrix:
    scatters[i,j] = scatter of the sequence with starting frame i and ending frame j
    """
    n = K.shape[0]
    # Prefix sum of diagonal elements (with a leading zero)
    diag = np.diag(K)
    K1 = np.concatenate(([0], diag)).cumsum()

    # 2D cumulative sum
    K2 = np.zeros((n + 1, n + 1), dtype=float)
    K2[1:, 1:] = np.cumsum(np.cumsum(K, axis=0), axis=1)

    # Allocate scatter matrix
    scatters = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            # Compute scatter for segment [i, j]
            seg_len = j - i + 1
            scatters[i, j] = (
                K1[j + 1] - K1[i]
                - (K2[j + 1, j + 1] + K2[i, i] - K2[j + 1, i] - K2[i, j + 1]) / seg_len
            )
    return scatters


def cpd_nonlin(
    K: np.ndarray,
    ncp: int,
    lmin: int = 1,
    lmax: int = 100000,
    backtrack: bool = True,
    verbose: bool = True,
    out_scatters: np.ndarray = None
) -> tuple:
    """
    Change point detection with dynamic programming.

    Args:
        K: Square kernel (Gram) matrix of shape (n, n).
        ncp: Number of change points to detect (must be >= 0).
        lmin: Minimum segment length (>= 1).
        lmax: Maximum segment length.
        backtrack: If True, returns the change point indices; otherwise only objective scores.
        verbose: If True, prints progress messages.
        out_scatters: Optional array to store the precomputed scatter matrix at index 0.

    Returns:
        cps: Array of detected change points of length ncp.
        scores: Objective scores for 0..ncp change points (length ncp+1).
    """
    m = int(ncp)  # ensure Python int
    n, n1 = K.shape
    if n != n1:
        raise ValueError("Kernel matrix must be square")
    if not (n >= (m + 1) * lmin and n <= (m + 1) * lmax):
        raise ValueError("Sequence length incompatible with ncp, lmin, and lmax")
    if verbose:
        print("Precomputing scatters...")
    J = calc_scatters(K)
    if out_scatters is not None:
        out_scatters[0] = J
    if verbose:
        print("Inferring best change points...")

    # Initialize dynamic programming table
    I = np.full((m + 1, n + 1), np.inf, dtype=float)
    # Base case: 0 change points over first l frames
    I[0, lmin:lmax] = J[0, (lmin - 1):lmax]

    # Backtracking pointers
    if backtrack:
        p = np.zeros((m + 1, n + 1), dtype=int)
    else:
        p = None

    # Fill DP table
    for k in range(1, m + 1):
        for l in range((k + 1) * lmin, n + 1):
            best_val = np.inf
            best_t = 0
            # Try all possible previous change positions
            t_start = max(k * lmin, l - lmax)
            t_end = l - lmin
            for t in range(t_start, t_end + 1):
                val = I[k - 1, t] + J[t, l - 1]
                if val < best_val:
                    best_val = val
                    best_t = t
            I[k, l] = best_val
            if backtrack:
                p[k, l] = best_t

    # Recover change points
    cps = np.zeros(m, dtype=int)
    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k - 1] = p[k, cur]
            cur = cps[k - 1]

    # Objective scores for 0..m change points
    scores = I[:, n].copy()
    scores[np.isinf(scores)] = np.inf

    return cps, scores
