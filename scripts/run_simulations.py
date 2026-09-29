"""Monte Carlo comparison of Merton's hedge and the quadratic hedge.

Writes results/scenarios.csv (one row per scenario) and results/base_case_pnl.npz.
Both strategies are always evaluated on the same paths, and scenarios that share
the same true model reuse the same paths (common random numbers), so differences
between scenarios are not driven by sampling noise in the paths.

Usage: python scripts/run_simulations.py [--n-paths 100000] [--seed 7]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jdhedge import PAYOFFS, Model, hedge_pnl, simulate_paths, variance_reduction  # noqa: E402

T = 1.0
BASE = Model(sigma=0.20, lam=2.0, m=0.0, delta=0.30)
STRADDLE = PAYOFFS["ATM straddle"]
FINE_STEPS = 500          # finest grid; coarser grids are sub-sampled from it
BASE_STEPS = 20
BATCH = 25_000


def scenarios():
    """(group, label, true model, hedger's model, payoff name, n_steps, cost)."""
    out = []
    for n in (1, 5, 20, 100, 500):
        out.append(("frequency", f"{n} steps", BASE, BASE, "ATM straddle", n, 0.0))
    for ratio in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.5, 10.0, 20.0):
        true = BASE.replace(sigma=float(np.sqrt(BASE.lam * BASE.delta**2 / ratio)))
        out.append(("regime", f"jump share {ratio:g}", true, true, "ATM straddle", BASE_STEPS, 0.0))
    for name in PAYOFFS:
        out.append(("payoff", name, BASE, BASE, name, BASE_STEPS, 0.0))
    for c in (0.0, 0.001, 0.003, 0.01, 0.03, 0.1):
        out.append(("cost", f"c = {c:g}", BASE, BASE, "ATM straddle", BASE_STEPS, c))
    shifts = {"sigma": (-0.4, -0.2, 0.0, 0.2, 0.4),
              "lam": (-0.5, -0.25, 0.0, 0.25, 0.5),
              "delta": (-0.4, -0.2, 0.0, 0.2, 0.4)}
    for n in (5, 20, 100):
        for param, rel in shifts.items():
            for r in rel:
                model = BASE.replace(**{param: getattr(BASE, param) * (1 + r)})
                out.append((f"misspec_{param}", f"{param} {r:+.0%}, {n} steps",
                            BASE, model, "ATM straddle", n, 0.0))
    return out


def run(n_paths: int, seed: int) -> None:
    rows, base_pnl = [], {}
    all_scenarios = scenarios()
    true_models = list(dict.fromkeys(s[2] for s in all_scenarios))
    start = time.time()
    for true in true_models:
        group = [s for s in all_scenarios if s[2] == true]
        steps = FINE_STEPS if true == BASE else BASE_STEPS
        pnl = [{"merton": [], "quadratic": []} for _ in group]
        traded = [{"merton": [], "quadratic": []} for _ in group]
        rng = np.random.default_rng(seed)
        for b in range(0, n_paths, BATCH):
            paths = simulate_paths(true, T, steps, min(BATCH, n_paths - b), rng)
            for i, (_, _, _, model, payoff, n_steps, cost) in enumerate(group):
                sub = paths[:, :: steps // n_steps]
                for strategy in ("merton", "quadratic"):
                    res = hedge_pnl(sub, T, model, PAYOFFS[payoff], strategy, cost)
                    pnl[i][strategy].append(res["pnl"])
                    traded[i][strategy].append(res["traded"])
        for i, (grp, label, _, model, payoff, n_steps, cost) in enumerate(group):
            pm = np.concatenate(pnl[i]["merton"])
            pq = np.concatenate(pnl[i]["quadratic"])
            vr, se = variance_reduction(pm, pq)
            row = dict(group=grp, label=label, payoff=payoff, n_steps=n_steps, cost=cost,
                       sigma_true=true.sigma, lam_true=true.lam, delta_true=true.delta,
                       jump_share=true.jump_share, sigma_model=model.sigma,
                       lam_model=model.lam, delta_model=model.delta, n_paths=len(pm),
                       var_reduction=vr, var_reduction_se=se)
            for tag, p in (("merton", pm), ("quadratic", pq)):
                row[f"std_{tag}"] = p.std(ddof=1)
                row[f"mean_{tag}"] = p.mean()
                row[f"q01_{tag}"] = np.quantile(p, 0.01)
                row[f"kurtosis_{tag}"] = kurtosis(p)
                row[f"traded_{tag}"] = np.concatenate(traded[i][tag]).mean()
            rows.append(row)
            if grp == "frequency" and n_steps == BASE_STEPS:
                base_pnl = {"merton": pm.astype(np.float32), "quadratic": pq.astype(np.float32)}
            print(f"{time.time() - start:7.0f}s  {grp:15s} {label:28s} "
                  f"var. reduction {100 * vr:6.2f}% +/- {196 * se:.2f}", flush=True)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "scenarios.csv", index=False)
    np.savez_compressed(out / "base_case_pnl.npz", **base_pnl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-paths", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(args.n_paths, args.seed)
