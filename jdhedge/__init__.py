"""Pricing and hedging of options under a Bachelier jump-diffusion."""

from .model import Model, Payoff, PAYOFFS
from .pricing import value_and_delta, quadratic_hedge, terminal_payoff
from .paths import simulate_paths
from .hedging import hedge_pnl, variance_reduction

__all__ = [
    "Model", "Payoff", "PAYOFFS",
    "value_and_delta", "quadratic_hedge", "terminal_payoff",
    "simulate_paths", "hedge_pnl", "variance_reduction",
]
