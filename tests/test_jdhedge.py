import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from jdhedge import (PAYOFFS, Model, hedge_pnl, quadratic_hedge, simulate_paths,
                     terminal_payoff, value_and_delta, variance_reduction)

T = 1.0
MODEL = Model(sigma=0.20, lam=2.0, m=0.10, delta=0.30)
X = np.linspace(-1.0, 1.0, 21)


def test_no_jumps_matches_bachelier():
    model = MODEL.replace(lam=0.0)
    call = ((1.0, "call", 0.2),)
    value, delta = value_and_delta(0.3, X, T, model, call)
    s = model.sigma * np.sqrt(T - 0.3)
    d = (X - 0.2) / s
    np.testing.assert_allclose(value, (X - 0.2) * norm.cdf(d) + s * norm.pdf(d), atol=1e-14)
    np.testing.assert_allclose(delta, norm.cdf(d), atol=1e-14)
    np.testing.assert_allclose(quadratic_hedge(0.3, X, T, model, call), delta, atol=1e-14)


def test_put_call_parity():
    call, put = ((1.0, "call", 0.1),), ((1.0, "put", 0.1),)
    c, dc = value_and_delta(0.2, X, T, MODEL, call)
    p, dp = value_and_delta(0.2, X, T, MODEL, put)
    np.testing.assert_allclose(c - p, X - 0.1, atol=1e-12)
    np.testing.assert_allclose(dc - dp, 1.0, atol=1e-12)


@pytest.mark.parametrize("name", PAYOFFS)
def test_quadratic_hedge_matches_quadrature(name):
    payoff, t = PAYOFFS[name], 0.4
    density = norm(MODEL.m, MODEL.delta).pdf
    for x0 in (-0.4, 0.05, 0.5):
        v0, vx = value_and_delta(t, x0, T, MODEL, payoff)
        jump = integrate.quad(lambda y: (value_and_delta(t, x0 + y, T, MODEL, payoff)[0] - v0)
                              * y * density(y), -3, 3, limit=200)[0]
        expected = (MODEL.sigma**2 * vx + MODEL.lam * jump) / MODEL.variance_rate
        assert quadratic_hedge(t, x0, T, MODEL, payoff) == pytest.approx(expected, abs=1e-9)


def test_truncation_is_negligible():
    payoff = PAYOFFS["W-profile"]
    loose = quadratic_hedge(0.1, X, T, MODEL, payoff, tol=1e-6)
    tight = quadratic_hedge(0.1, X, T, MODEL, payoff, tol=1e-14)
    np.testing.assert_allclose(loose, tight, atol=1e-7)


def test_paths_have_the_right_moments():
    paths = simulate_paths(MODEL, T, 4, 400_000, np.random.default_rng(0))
    increments = np.diff(paths, axis=1)
    se_mean = np.sqrt(MODEL.variance_rate * T / 4 / len(paths))
    assert np.all(np.abs(increments.mean(axis=0)) < 5 * se_mean)
    np.testing.assert_allclose(paths[:, -1].var(), MODEL.variance_rate * T, rtol=0.01)


@pytest.mark.parametrize("name", PAYOFFS)
def test_price_matches_monte_carlo(name):
    payoff = PAYOFFS[name]
    h = terminal_payoff(simulate_paths(MODEL, T, 1, 1_000_000, np.random.default_rng(1))[:, -1], payoff)
    exact = value_and_delta(0.0, 0.0, T, MODEL, payoff)[0]
    assert abs(h.mean() - exact) < 5 * h.std() / np.sqrt(len(h))


def test_hedged_pnl_is_unbiased_and_hedging_helps():
    paths = simulate_paths(MODEL, T, 50, 50_000, np.random.default_rng(2))
    payoff = PAYOFFS["ATM straddle"]
    unhedged = value_and_delta(0.0, 0.0, T, MODEL, payoff)[0] - terminal_payoff(paths[:, -1], payoff)
    for strategy in ("merton", "quadratic"):
        pnl = hedge_pnl(paths, T, MODEL, payoff, strategy)["pnl"]
        assert abs(pnl.mean()) < 5 * pnl.std() / np.sqrt(len(pnl))
        assert pnl.var() < 0.75 * unhedged.var()  # jumps limit what any hedge can remove


def test_variance_reduction_standard_error():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(20_000)
    b = 0.8 * a + 0.3 * rng.standard_normal(20_000)
    vr, se = variance_reduction(a, b)
    boot = []
    for _ in range(400):
        i = rng.integers(0, len(a), len(a))
        boot.append(1 - b[i].var() / a[i].var())
    assert se == pytest.approx(np.std(boot), rel=0.15)
