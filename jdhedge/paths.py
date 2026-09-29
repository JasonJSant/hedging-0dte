"""Exact simulation of the jump-diffusion on a uniform time grid."""

import numpy as np

from .model import Model


def simulate_paths(model: Model, T: float, n_steps: int, n_paths: int,
                   rng: np.random.Generator, x0: float = 0.0) -> np.ndarray:
    """Paths of X on t_k = k T / n_steps, shape (n_paths, n_steps + 1).

    Over each step, the number of jumps k is Poisson(lam dt) and, given k, their
    sum is exactly N(k m, k delta^2), so the grid values carry no discretization error.
    """
    dt = T / n_steps
    shape = (n_paths, n_steps)
    diffusion = model.sigma * np.sqrt(dt) * rng.standard_normal(shape)
    k = rng.poisson(model.lam * dt, shape)
    jumps = k * model.m + np.sqrt(k) * model.delta * rng.standard_normal(shape)
    increments = diffusion + jumps - model.lam * model.m * dt
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    np.cumsum(increments, axis=1, out=paths[:, 1:])
    paths[:, 1:] += x0
    return paths
