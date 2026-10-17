"""Experiment: fair root-cause benchmark (offline; does not change the live demo).

Injects a fault into a known joint k on healthy UR3 data, propagates it to two "victim" joints,
and asks each method to name the root. Six scenarios, from easy to deliberately hard, so the
result cannot be called circular (the clean scenario uses exactly the lag structure Granger looks for):

  S1 clean        victims react with fixed lags 1 s / 2 s                      (original test)
  S2 mixed lags   random lags 1-4 s, random gains, extra sensor noise on victims
  S3 weak fault   half-strength fault, weak coupling
  S4 nonlinear    victims respond through a saturating (tanh) nonlinearity, lags 1-3 s
  S5 confounded   a shared load disturbance drives ALL joints at once, plus the fault (as S2)
  S6 same-second  victims react within the same 1-s sample (lag 0) - no temporal order visible

Methods (all receive the same 60-s window of joint currents):
  Granger (ours)        causal.causal_summary  (net outgoing - incoming significant Granger strength)
  Liang-Kleeman IF      information flow T_{j->i} (Liang 2014), root = argmax(out - in) of |T|
                        - the causal measure used by the dataset authors (Tyrovolas et al., 2024)
  Most abnormal         the joint whose current deviates most from its healthy baseline
  Correlation           the joint most correlated with all others (no direction)
Reported: accuracy with 95% Wilson confidence interval over N trials per scenario.

Run:  python exp_root_cause.py
"""
import json

import numpy as np
import pandas as pd

from src import causal as C
from src.data import N_JOINTS, ROOT, col, load_clean
from src.pipeline import make_baseline

N_TRIALS = 100
SEED = 7
OUT = ROOT / "outputs"


def liang_kleeman(X, dt=1.0):
    """Bivariate Liang-Kleeman information flow, T[j, i] = flow from j to i (Liang, PRE 2014)."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    x, dx = X[:-1], (X[1:] - X[:-1]) / dt
    N = X.shape[1]
    T = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            c = np.cov(np.vstack([x[:, i], x[:, j]]))
            cii, cij, cjj = c[0, 0], c[0, 1], c[1, 1]
            c_i_di = np.cov(x[:, i], dx[:, i])[0, 1]
            c_j_di = np.cov(x[:, j], dx[:, i])[0, 1]
            den = cii ** 2 * cjj - cii * cij ** 2
            T[j, i] = (cii * cij * c_j_di - cij ** 2 * c_i_di) / den if abs(den) > 1e-12 else 0.0
    return T


def root_from_matrix(M):
    A = np.abs(M)
    np.fill_diagonal(A, 0)
    return int(np.argmax(A.sum(1) - A.sum(0)))


def methods(w, baseline):
    cols = [col("current", j) for j in range(N_JOINTS)]
    X = w[cols].to_numpy(float)
    g = C.causal_summary(w, baseline)["root_joint"]  # live causal layer (lag order C.LAG = 4)
    G2, P2 = C.granger_matrix(X, L=2)
    g2 = root_from_matrix(G2 * (P2 < C.ALPHA))      # earlier version with 2-s lags, for comparison
    lk = root_from_matrix(liang_kleeman(X))
    z = np.array([np.mean(np.abs((X[:, j] - baseline["mean"][cols[j]]) / (baseline["std"][cols[j]] + 1e-6)))
                  for j in range(N_JOINTS)])
    corr = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(corr, 0)
    return {"Granger (ours)": -1 if g is None else g, "Granger L=2": g2, "Liang-Kleeman IF": lk,
            "Most abnormal": int(np.argmax(z)), "Correlation": int(np.argmax(corr.sum(0)))}


def lagged(x, lag):
    return x if lag == 0 else np.r_[np.zeros(lag), x[:-lag]]


def inject(w, k, scenario, baseline, rng):
    W = C.WINDOW
    sd = lambda j: baseline["std"][col("current", j)]
    strength = 0.5 if scenario == "S3 weak fault" else 1.0
    burst = (rng.normal(0, 1.5 * sd(k), W) + np.linspace(0, 2 * sd(k), W)) * strength
    w[col("current", k)] += burst
    victims = rng.choice([j for j in range(N_JOINTS) if j != k], size=2, replace=False)
    for n, v in enumerate(victims):
        if scenario == "S1 clean":
            lag, gain = n + 1, (0.8, 0.6)[n]
        elif scenario == "S3 weak fault":
            lag, gain = n + 1, rng.uniform(0.4, 0.6)
        elif scenario == "S6 same-second":
            lag, gain = 0, rng.uniform(0.4, 0.9)
        elif scenario == "S7 victims louder":
            # Compensating joints draw MORE current than the faulty one (e.g. a stiff wrist makes the
            # heavily loaded shoulder work much harder): the "most abnormal" joint is a victim.
            lag, gain = int(rng.integers(1, 4)), rng.uniform(1.8, 2.8)
        else:
            lag, gain = int(rng.integers(1, 5 if scenario != "S4 nonlinear" else 4)), rng.uniform(0.3, 0.9)
        drive = lagged(burst, lag) / sd(k)
        if scenario == "S4 nonlinear":
            drive = 2.0 * np.tanh(drive / 2.0)
        w[col("current", v)] += gain * drive * sd(v)
        if scenario in ("S2 mixed lags", "S5 confounded", "S7 victims louder"):
            w[col("current", v)] += rng.normal(0, 0.5 * sd(v), W)
    if scenario == "S5 confounded":  # shared payload / trajectory disturbance hitting every joint at once
        d = np.cumsum(rng.normal(0, 0.3, W))
        for j in range(N_JOINTS):
            w[col("current", j)] += rng.uniform(0.5, 1.0) * d * sd(j)
    w[col("temp", k)] += np.linspace(0, 3, W)
    return w


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def main():
    df = load_clean()
    i_tr = int(len(df) * 0.55)
    tr = df.iloc[:i_tr]
    normal = tr[tr["fault"] == 0].reset_index(drop=True)
    baseline = make_baseline(normal)
    rng = np.random.default_rng(SEED)
    scenarios = ["S1 clean", "S2 mixed lags", "S3 weak fault", "S4 nonlinear", "S5 confounded", "S6 same-second",
                 "S7 victims louder"]
    rows = []
    for sc in scenarios:
        hits = {}
        for _ in range(N_TRIALS):
            s = int(rng.integers(0, len(normal) - C.WINDOW))
            w = normal.iloc[s:s + C.WINDOW].copy().reset_index(drop=True)
            k = int(rng.integers(0, N_JOINTS))
            w = inject(w, k, sc, baseline, rng)
            for m, pred in methods(w, baseline).items():
                hits[m] = hits.get(m, 0) + int(pred == k)
        for m, h in hits.items():
            lo, hi = wilson(h, N_TRIALS)
            rows.append({"scenario": sc, "method": m, "accuracy": h / N_TRIALS, "ci_low": lo, "ci_high": hi})
        print(sc, {m: f"{h / N_TRIALS:.2f}" for m, h in hits.items()})

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "exp_root_cause.csv", index=False)
    table = res.pivot(index="scenario", columns="method", values="accuracy").reindex(scenarios)
    print("\nAccuracy (chance = %.2f), %d trials per scenario:\n" % (1 / N_JOINTS, N_TRIALS))
    print(table.round(2).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["Granger (ours)", "Granger L=2", "Liang-Kleeman IF", "Most abnormal", "Correlation"]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(scenarios))
    for n, m in enumerate(order):
        r = res[res.method == m].set_index("scenario").reindex(scenarios)
        err = [np.clip(r.accuracy - r.ci_low, 0, None), np.clip(r.ci_high - r.accuracy, 0, None)]
        ax.bar(x + (n - 2) * 0.16, r.accuracy, 0.16, label=m, yerr=err, capsize=2)
    ax.axhline(1 / N_JOINTS, color="k", ls="--", lw=0.8, label="chance")
    ax.set(xticks=x, xticklabels=scenarios, ylim=(0, 1.05), ylabel="root-cause accuracy",
           title=f"Root-cause identification across fault scenarios ({N_TRIALS} trials each, 95% CI)")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "root_cause_scenarios.png", dpi=150)


if __name__ == "__main__":
    main()
