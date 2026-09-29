"""Discrete hedging of a short option position along simulated paths."""

import numpy as np

from .model import Model, Payoff
from .pricing import quadratic_hedge, terminal_payoff, value_and_delta


def hedge_pnl(paths: np.ndarray, T: float, model: Model, payoff: Payoff,
              strategy: str, cost: float = 0.0) -> dict[str, np.ndarray]:
    """Terminal P&L of selling the option at its model value and hedging.

    The hedge is set at each grid time t_0, ..., t_{n-1} and held until the next one:

        PnL = V_0 - H(X_T) + sum_k phi_k (X_{t_{k+1}} - X_{t_k}) - cost * sum_k |phi_k - phi_{k-1}|,

    with phi_{-1} = phi_n = 0, so the initial trade and the final unwind are both charged.
    `model` is the hedger's model; it may differ from the one that generated the paths.
    """
    if strategy not in ("merton", "quadratic"):
        raise ValueError("strategy must be 'merton' or 'quadratic'")
    n_paths, n_points = paths.shape
    n_steps = n_points - 1
    times = np.linspace(0.0, T, n_points)

    positions = np.zeros((n_paths, n_steps + 2))  # phi_{-1}, phi_0, ..., phi_{n-1}, phi_n
    for k in range(n_steps):
        x = paths[:, k]
        if strategy == "merton":
            positions[:, k + 1] = value_and_delta(times[k], x, T, model, payoff)[1]
        else:
            positions[:, k + 1] = quadratic_hedge(times[k], x, T, model, payoff)

    v0 = value_and_delta(0.0, paths[0, 0], T, model, payoff)[0]
    hedge_gain = (positions[:, 1:-1] * np.diff(paths, axis=1)).sum(axis=1)
    traded = np.abs(np.diff(positions, axis=1)).sum(axis=1)
    gross = v0 - terminal_payoff(paths[:, -1], payoff) + hedge_gain
    return {"pnl": gross - cost * traded, "gross": gross, "traded": traded}


def variance_reduction(pnl_ref: np.ndarray, pnl_new: np.ndarray) -> tuple[float, float]:
    """1 - Var(pnl_new) / Var(pnl_ref) and its standard error.

    Both P&L samples come from the same paths, so the error is computed with the
    delta method on the paired squared deviations.
    """
    u = (pnl_ref - pnl_ref.mean()) ** 2
    w = (pnl_new - pnl_new.mean()) ** 2
    ratio = w.mean() / u.mean()
    se = np.std(w - ratio * u, ddof=1) / (u.mean() * np.sqrt(len(u)))
    return 1.0 - ratio, se
