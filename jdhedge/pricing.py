"""Closed-form prices and hedge ratios for call/put portfolios.

Conditioning on the number n of jumps over the remaining time theta = T - t,
X_T - X_t is Gaussian with mean mu_n = (n - lam theta) m and variance
v_n = sigma^2 theta + n delta^2. Prices are therefore Poisson mixtures of
Bachelier prices:

    V(t, x) = sum_n p_n Pi_n(t, x),    p_n = P(N_theta = n).

The quadratic (variance-optimal) hedge is

    phi*(t, x) = [sigma^2 V_x + lam E[(V(t, x+Y) - V(t, x)) Y]] / q.

For Gaussian jumps, Stein's lemma gives E[g(x+Y) Y] = m E[g(x+Y)] + delta^2 E[g'(x+Y)],
and averaging Pi_n over one extra jump gives Pi_{n+1}. Hence

    E[(V(t, x+Y) - V(t, x)) Y] = sum_n p_n [m (Pi_{n+1} - Pi_n) + delta^2 Delta_{n+1}],

where Delta_n = d Pi_n / dx. All functions are vectorized over x.
"""

import numpy as np
from scipy.special import ndtr
from scipy.stats import poisson

from .model import Model, Payoff

_SQRT_2PI = np.sqrt(2.0 * np.pi)
_VAR_FLOOR = 1e-14


def poisson_weights(mean: float, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Jump counts n = 0..n_max and their Poisson probabilities, truncated so
    that the omitted tail mass is below tol."""
    if mean <= 0.0:
        return np.array([0]), np.array([1.0])
    n_max = max(int(poisson.ppf(1.0 - tol, mean)), int(np.ceil(mean + 10.0 * np.sqrt(mean + 1.0))))
    n = np.arange(n_max + 1)
    return n, poisson.pmf(n, mean)


def _bachelier(a: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """E[(a + sqrt(v) Z)^+] and its derivative in a, for Z ~ N(0, 1)."""
    s = np.sqrt(np.maximum(v, _VAR_FLOOR))
    d = a / s
    return a * ndtr(d) + s * np.exp(-0.5 * d * d) / _SQRT_2PI, ndtr(d)


def components(x: np.ndarray, n: np.ndarray, theta: float, model: Model,
                payoff: Payoff) -> tuple[np.ndarray, np.ndarray]:
    """Pi_n(t, x) and Delta_n(t, x), with shape (len(n),) + x.shape."""
    mu = ((n - model.lam * theta) * model.m).reshape((-1,) + (1,) * x.ndim)
    v = (model.sigma**2 * theta + n * model.delta**2).reshape(mu.shape)
    price = np.zeros(mu.shape[:1] + x.shape)
    delta = np.zeros_like(price)
    for weight, kind, strike in payoff:
        if kind == "call":
            p, d = _bachelier(x + mu - strike, v)
            price += weight * p
            delta += weight * d
        elif kind == "put":
            p, d = _bachelier(strike - x - mu, v)
            price += weight * p
            delta -= weight * d
        else:
            raise ValueError(f"unknown instrument kind: {kind!r}")
    return price, delta


def terminal_payoff(x, payoff: Payoff) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for weight, kind, strike in payoff:
        if kind == "call":
            out += weight * np.maximum(x - strike, 0.0)
        elif kind == "put":
            out += weight * np.maximum(strike - x, 0.0)
        else:
            raise ValueError(f"unknown instrument kind: {kind!r}")
    return out


def value_and_delta(t: float, x, T: float, model: Model, payoff: Payoff,
                    tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Option value V(t, x) and Merton's hedge V_x(t, x)."""
    x = np.asarray(x, dtype=float)
    theta = T - t
    n, p = poisson_weights(model.lam * theta, tol)
    price, delta = components(x, n, theta, model, payoff)
    w = p.reshape((-1,) + (1,) * x.ndim)
    return (w * price).sum(axis=0), (w * delta).sum(axis=0)


def quadratic_hedge(t: float, x, T: float, model: Model, payoff: Payoff,
                    tol: float = 1e-12) -> np.ndarray:
    """Variance-optimal hedge ratio phi*(t, x)."""
    x = np.asarray(x, dtype=float)
    theta = T - t
    n, p = poisson_weights(model.lam * theta, tol)
    # Components for n = 0..n_max+1 so that Pi_{n+1} is available for every n.
    price, delta = components(x, np.append(n, n[-1] + 1), theta, model, payoff)
    w = p.reshape((-1,) + (1,) * x.ndim)
    v_x = (w * delta[:-1]).sum(axis=0)
    jump_term = (w * (model.m * (price[1:] - price[:-1]) + model.delta**2 * delta[1:])).sum(axis=0)
    return (model.sigma**2 * v_x + model.lam * jump_term) / model.variance_rate
