"""
Anti-overfit / data-snooping validation (research-backed).

  - PBO via Combinatorially Symmetric Cross-Validation (Bailey, Borwein,
    Lopez de Prado, Zhu 2017): probability the in-sample-best config underperforms
    the OOS median. Hold-out is unreliable; CSCV averages over many IS/OOS splits.
  - Hansen's Superior Predictive Ability (SPA) test (2005): data-snooping-robust
    test that a strategy beats a benchmark across many alternatives; more powerful
    and less sensitive to poor alternatives than White's Reality Check.

PBO interpretation: > 0.5 means the selection is likely overfit.
SPA interpretation: low p-value means superiority survives data-snooping control.
"""

from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd


def pbo_cscv(perf_matrix: pd.DataFrame, S: int = 8) -> float:
    """perf_matrix: T periods x N config returns. Returns PBO in [0,1].
    Best in-sample config's OOS relative rank -> logit; PBO = P(logit <= 0)."""
    M = perf_matrix.dropna()
    T, N = M.shape
    if N < 2 or T < S:
        return float("nan")
    groups = np.array_split(np.arange(T), S)
    logits = []
    for c in combinations(range(S), S // 2):
        is_rows = np.concatenate([groups[i] for i in c])
        oos_rows = np.concatenate([groups[i] for i in range(S) if i not in c])
        perf_is = M.iloc[is_rows].mean()
        perf_oos = M.iloc[oos_rows].mean()
        n_star = int(np.argmax(perf_is.values))                  # best IS config
        oos_rank = perf_oos.rank().iloc[n_star]                  # 1..N (higher better)
        w = oos_rank / (N + 1)                                   # relative rank in (0,1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return float(np.mean(logits <= 0))                           # P(best IS below OOS median)


def spa_pvalue(strategy_returns: pd.Series, benchmark_returns: pd.Series,
               reps: int = 1000) -> float:
    """Hansen SPA: is the strategy superior to the benchmark after snooping
    control? Uses negative returns as 'loss' (SPA tests benchmark vs models on
    loss). Returns the consistent p-value; low => superiority is real."""
    try:
        from arch.bootstrap import SPA
        a, b = strategy_returns.align(benchmark_returns, join="inner")
        a, b = a.dropna(), b.dropna()
        a, b = a.align(b, join="inner")
        if len(a) < 30:
            return float("nan")
        # arch SPA: benchmark losses vs model losses (lower loss = better).
        bench_loss = -b.values
        model_loss = (-a.values).reshape(-1, 1)
        spa = SPA(bench_loss, model_loss, reps=reps, seed=42)
        spa.compute()
        pv = spa.pvalues
        return float(pv["consistent"]) if hasattr(pv, "__getitem__") else float(pv[1])
    except Exception as e:
        print(f"SPA unavailable ({e})")
        return float("nan")


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    # Overfit case: pure noise configs -> PBO should be near 0.5 (no real edge)
    noise = pd.DataFrame(rng.normal(0, 0.01, (240, 20)))
    print("PBO (pure noise, expect ~0.5):", round(pbo_cscv(noise), 3))
    # One genuinely superior config -> PBO should drop well below 0.5
    edge = noise.copy(); edge[0] += 0.004
    print("PBO (one real edge, expect <0.5):", round(pbo_cscv(edge), 3))
