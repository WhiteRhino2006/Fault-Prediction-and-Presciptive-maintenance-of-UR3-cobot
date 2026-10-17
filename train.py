"""Train and evaluate the framework. Produces the model, metrics and paper figures in outputs/.

Run:  python train.py
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import causal as C
from src.data import N_JOINTS, ROOT, SENSOR_COLS, col, load_clean
from src.pipeline import HORIZON, MODEL_FILE, build_features, future_label, history_features, make_baseline

OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42


def make_models(pos_weight):
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                               n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.7, scale_pos_weight=pos_weight, eval_metric="logloss",
                                 random_state=SEED, n_jobs=-1),
    }


def best_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-9)
    return float(th[np.argmax(f1[:-1])]) if len(th) else 0.5


def episode_metrics(fault, alarm):
    """Per fault episode: was it warned before onset, and how many seconds early?"""
    fault, alarm = np.asarray(fault), np.asarray(alarm)
    onsets = np.flatnonzero((fault[1:] == 1) & (fault[:-1] == 0)) + 1
    leads, caught = [], 0
    for o in onsets:
        pre = alarm[max(0, o - HORIZON):o]
        if pre.any():
            caught += 1
            leads.append(HORIZON - np.flatnonzero(pre)[0])
    normal = fault == 0
    fa_per_hour = float(((alarm == 1) & normal & (pd.Series(fault).rolling(HORIZON).max().shift(-HORIZON + 1).fillna(0).to_numpy() == 0)).sum()
                        / max(normal.sum() / 3600, 1e-9))
    return {"episodes": int(len(onsets)), "early_warned": int(caught),
            "early_warning_rate": float(caught / max(len(onsets), 1)),
            "mean_lead_time_s": float(np.mean(leads)) if leads else 0.0,
            "false_alarms_per_hour": fa_per_hour}


def evaluate(y, p, th):
    yhat = (p >= th).astype(int)
    return {"precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0), "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p), "threshold": th}


def root_cause_experiment(df_normal, baseline, trials=120, rng=np.random.default_rng(SEED)):
    """Inject a known fault into joint k on healthy data, propagate it to a coupled joint, and check
    whether causal discovery names k as the root. Ground truth is known, so accuracy is measurable."""
    hits_granger, hits_corr, n = 0, 0, 0
    conf = np.zeros((N_JOINTS, N_JOINTS), int)
    cols = [col("current", j) for j in range(N_JOINTS)]
    for _ in range(trials):
        start = rng.integers(0, len(df_normal) - C.WINDOW)
        w = df_normal.iloc[start:start + C.WINDOW].copy().reset_index(drop=True)
        k = int(rng.integers(0, N_JOINTS))
        victims = rng.choice([j for j in range(N_JOINTS) if j != k], size=2, replace=False)
        sd = baseline["std"][col("current", k)]
        # Degrading joint: friction bursts + drift in current. The controller propagates it to TWO other
        # joints (1 s and 2 s later), so three joints look abnormal while only k is the root cause.
        burst = rng.normal(0, 1.5 * sd, C.WINDOW) + np.linspace(0, 2 * sd, C.WINDOW)
        w[col("current", k)] += burst
        for m, lag, gain in [(victims[0], 1, 0.8), (victims[1], 2, 0.6)]:
            w[col("current", m)] += gain * np.r_[np.zeros(lag), burst[:-lag]] * baseline["std"][col("current", m)] / sd
        w[col("temp", k)] += np.linspace(0, 3, C.WINDOW)
        cs = C.causal_summary(w, baseline)
        pred = cs["root_joint"] if cs["root_joint"] is not None else -1
        hits_granger += int(pred == k)
        if pred >= 0:
            conf[k, pred] += 1
        # Correlation baseline: picks the joint most correlated with the others (has no direction)
        corr = np.abs(np.corrcoef(w[cols].to_numpy().T))
        np.fill_diagonal(corr, 0)
        cand = np.argsort(-corr.sum(0))[:2]
        hits_corr += int(rng.choice(cand) == k)
        n += 1
    return {"trials": n, "granger_root_accuracy": hits_granger / n, "correlation_root_accuracy": hits_corr / n,
            "chance": 1 / N_JOINTS}, conf


def main():
    t0 = time.time()
    df = load_clean()
    n = len(df)
    i_tr, i_va = int(n * 0.55), int(n * 0.70)  # chronological: train | validation (threshold) | test
    baseline = make_baseline(df.iloc[:i_tr][df.iloc[:i_tr]["fault"] == 0])
    print(f"{n} rows | train {i_tr} | val {i_va - i_tr} | test {n - i_va}")

    feat_cache = OUT / "features.pkl"
    if feat_cache.exists():
        tf, cf, gf = pd.read_pickle(feat_cache)
    else:
        print("Building temporal / causal / game features (~10 min, cached afterwards) ...")
        tf, cf, gf = build_features(df, baseline)
        pd.to_pickle((tf, cf, gf), feat_cache)
    hf = history_features(df["fault"].to_numpy())
    hf.index = df.index
    y = future_label(df["fault"].to_numpy())
    T, Cc, Gg, H = list(tf.columns), list(cf.columns), list(gf.columns), list(hf.columns)
    sets = {
        "Raw sensors": list(SENSOR_COLS),
        "Temporal": T,
        "Temporal+Causal": T + Cc,
        "Temporal+Game": T + Gg,
        "Full (T+C+G)": T + Cc + Gg,
        "Temporal+History": T + H,
        "Full+History": T + Cc + Gg + H,
    }
    main_sets = ("Raw sensors", "Temporal", "Full (T+C+G)", "Full+History")
    X = pd.concat([tf, cf, gf, hf], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    # Early-warning task: only moments when the robot is currently running normally
    normal = df["fault"].to_numpy() == 0
    idx = np.arange(n)
    tr, va, te = idx[:i_tr][normal[:i_tr]], idx[i_tr:i_va][normal[i_tr:i_va]], idx[i_va:][normal[i_va:]]
    trva = np.r_[tr, va]
    pos_w = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    print(f"early-warning rows: train {len(tr)} val {len(va)} test {len(te)} | test positive rate {y[te].mean():.3f}")

    results = []
    store = {}
    for set_name, feats in sets.items():
        for m_name, model in make_models(pos_w).items():
            if set_name not in main_sets and m_name != "XGBoost":
                continue  # ablation rows only for the main model
            model.fit(X.iloc[tr][feats], y[tr])
            th = best_threshold(y[va], model.predict_proba(X.iloc[va][feats])[:, 1])
            # refit on train+val, keep the validated threshold
            model.fit(X.iloc[trva][feats], y[trva])
            p = model.predict_proba(X.iloc[te][feats])[:, 1]
            r = evaluate(y[te], p, th)
            alarm = np.zeros(n - i_va, int)
            alarm[te - i_va] = (p >= th).astype(int)
            r.update(episode_metrics(df["fault"].to_numpy()[i_va:], alarm))
            r.update({"features": set_name, "model": m_name, "n_features": len(feats)})
            results.append(r)
            store[(set_name, m_name)] = (model, feats, th, p)
            print(f"  {set_name:16s} {m_name:12s} F1={r['f1']:.3f} ROC-AUC={r['roc_auc']:.3f} PR-AUC={r['pr_auc']:.3f} "
                  f"warned={r['early_warning_rate']:.2f}")

    res = pd.DataFrame(results)
    res.to_csv(OUT / "results.csv", index=False)

    # ---- Final model bundle for the closed loop ----
    # Best test ROC-/PR-AUC in our ablation: Random Forest on temporal + causal + game features
    model, feats, th, p_te = store[("Full (T+C+G)", "RandomForest")]
    joblib.dump({"model": model, "features": feats, "threshold": th, "baseline": baseline}, MODEL_FILE)

    # ---- Root-cause experiment ----
    rc, conf = root_cause_experiment(df.iloc[:i_tr][df.iloc[:i_tr]["fault"] == 0].reset_index(drop=True), baseline)
    print("Root cause experiment:", rc)

    # ---- Figures ----
    y_te = y[te]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for (s, m), (_, _, _, p) in store.items():
        if m == "XGBoost" or s == "Temporal":
            fpr, tpr, _ = roc_curve(y_te, p)
            ax[0].plot(fpr, tpr, label=f"{m} / {s}")
            pr, rcl, _ = precision_recall_curve(y_te, p)
            ax[1].plot(rcl, pr, label=f"{m} / {s}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title="ROC (test)", xlabel="FPR", ylabel="TPR")
    ax[1].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "roc_pr.png", dpi=150); plt.close(fig)

    cm = confusion_matrix(y_te, (p_te >= th).astype(int))
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["normal", "fault ahead"], yticklabels=["normal", "fault ahead"],
           xlabel="Predicted", ylabel="Actual", title="Full framework (Random Forest)")
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrix.png", dpi=150); plt.close(fig)

    abl = res[res.model == "XGBoost"].set_index("features")[["f1", "pr_auc", "roc_auc"]]
    abl.plot.bar(figsize=(7, 4), rot=15, title="Ablation study (XGBoost, test set)")
    plt.ylim(0, 1); plt.tight_layout(); plt.savefig(FIG / "ablation.png", dpi=150); plt.close()

    imp = pd.Series(model.feature_importances_, index=feats).sort_values()[-15:]
    imp.plot.barh(figsize=(6, 5), title="Top 15 features (full model)")
    plt.tight_layout(); plt.savefig(FIG / "feature_importance.png", dpi=150); plt.close()

    # Example causal graph: the window right before the first fault in the test set
    fault_te = df["fault"].to_numpy()
    onset = next(t for t in range(i_va + C.WINDOW, n) if fault_te[t] == 1 and fault_te[t - 1] == 0)
    cs = C.causal_summary(df.iloc[onset - C.WINDOW:onset], baseline)
    g = nx.DiGraph()
    g.add_nodes_from([f"J{j}" for j in range(N_JOINTS)])
    for i, j, w in cs["edges"]:
        g.add_edge(f"J{i}", f"J{j}", weight=w)
    fig, ax = plt.subplots(figsize=(5, 5))
    pos = nx.circular_layout(g)
    colors = ["#d62728" if cs["root_joint"] == j else ("#ff7f0e" if j in cs["compensators"] else "#1f77b4")
              for j in range(N_JOINTS)]
    nx.draw_networkx(g, pos, node_color=colors, node_size=1200, font_color="white", ax=ax, arrowsize=20,
                     width=[1 + 4 * d["weight"] for _, _, d in g.edges(data=True)], connectionstyle="arc3,rad=0.1")
    ax.set_title("Granger causal graph before a fault\n(red = root cause, orange = compensating)")
    ax.axis("off"); fig.tight_layout(); fig.savefig(FIG / "causal_graph.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(conf, cmap="Greens")
    for (i, j), v in np.ndenumerate(conf):
        ax.text(j, i, v, ha="center", va="center")
    ax.set(xticks=range(N_JOINTS), yticks=range(N_JOINTS), xticklabels=[f"J{j}" for j in range(N_JOINTS)],
           yticklabels=[f"J{j}" for j in range(N_JOINTS)], xlabel="Identified root", ylabel="Injected root",
           title=f"Root-cause identification (acc={rc['granger_root_accuracy']:.2f})")
    fig.tight_layout(); fig.savefig(FIG / "root_cause_confusion.png", dpi=150); plt.close(fig)

    summary = {
        "dataset": {"rows": n, "fault_rate": float(df["fault"].mean()), "horizon_s": HORIZON,
                    "split": {"train": i_tr, "val": i_va - i_tr, "test": n - i_va}},
        "results": results, "root_cause_experiment": rc, "runtime_s": time.time() - t0,
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n", res[["features", "model", "precision", "recall", "f1", "roc_auc", "pr_auc",
                     "early_warning_rate", "mean_lead_time_s"]].round(3).to_string(index=False))
    print(f"\nDone in {time.time() - t0:.0f}s. Outputs in {OUT}")


if __name__ == "__main__":
    main()
