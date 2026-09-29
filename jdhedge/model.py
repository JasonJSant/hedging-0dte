"""Model parameters and payoff definitions.

The underlying follows a Bachelier jump-diffusion under the pricing measure,

    X_t = X_0 + sigma W_t - lam m t + sum_{i <= N_t} Y_i,

with N a Poisson process of intensity lam and i.i.d. jumps Y_i ~ N(m, delta^2).
The drift -lam m t makes X a martingale.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Model:
    sigma: float  # Bachelier (absolute) volatility
    lam: float    # jump intensity
    m: float      # mean jump size
    delta: float  # jump size standard deviation

    @property
    def variance_rate(self) -> float:
        """q = sigma^2 + lam E[Y^2], the variance of X per unit time."""
        return self.sigma**2 + self.lam * (self.m**2 + self.delta**2)

    @property
    def jump_share(self) -> float:
        """lam E[Y^2] / sigma^2, the jump-to-diffusion variance ratio."""
        return self.lam * (self.m**2 + self.delta**2) / self.sigma**2

    def replace(self, **kwargs) -> "Model":
        return replace(self, **kwargs)


# A payoff is a portfolio of calls and puts: a tuple of (weight, kind, strike)
# with kind in {"call", "put"}.
Payoff = tuple[tuple[float, str, float], ...]


def w_profile(K: float = 0.30, alpha: float = 0.50) -> Payoff:
    return ((1.0, "put", -K), (-alpha, "call", 0.0), (-alpha, "put", 0.0), (1.0, "call", K))


PAYOFFS: dict[str, Payoff] = {
    "ATM straddle": ((1.0, "call", 0.0), (1.0, "put", 0.0)),
    "W-profile": w_profile(),
    "Risk reversal": ((1.0, "call", 0.30), (-1.0, "put", -0.30)),
    "Call ratio spread": ((3.0, "call", 0.30), (-1.0, "call", -0.30)),
}
