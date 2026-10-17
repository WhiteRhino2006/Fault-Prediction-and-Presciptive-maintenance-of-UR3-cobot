"""Train and evaluate the Causal-GAT and the plain-GAT ablation; adds rows to outputs/results.csv.

Run after train.py:  python train_gnn.py
Uses exactly the same split, label, normal-state filtering and threshold rule as train.py.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.data import N_JOINTS, ROOT, load_clean
from src.gnn import CausalGAT, granger_tensor, node_feature_columns
from src.pipeline import future_label, make_baseline
from train import FIG, OUT, best_threshold, episode_metrics, evaluate

SEEDS = [0, 1, 2]
EPOCHS = 60


def tensors(tf, G, rows, per_joint, glob, mu, sd, gmu, gsd):
    x = np.stack([(tf.iloc[rows][cols].to_numpy(np.float32) - mu) / sd for cols in per_joint], axis=1)
    g = (tf.iloc[rows][glob].to_numpy(np.float32) - gmu) / gsd
    return torch.tensor(x), torch.tensor(g), torch.tensor(G[rows])


def train_one(causal, seed, data, y, tr, va):
    torch.manual_seed(seed)
    np.random.seed(seed)
    xtr, gtr, Gtr = data(tr)
    xva, gva, Gva = data(va)
    ytr = torch.tensor(y[tr], dtype=torch.float32)
    model = CausalGAT(xtr.shape[-1], gtr.shape[-1], causal=causal)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    pos_w = ((ytr == 0).sum() / max((ytr == 1).sum(), 1)).float()
    best, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(ytr))
        for b in range(0, len(perm), 256):
            i = perm[b:b + 256]
            logit, _ = model(xtr[i], gtr[i], Gtr[i])
            loss = F.binary_cross_entropy_with_logits(logit, ytr[i], pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pva = torch.sigmoid(model(xva, gva, Gva)[0]).numpy()
        score = average_precision_score(y[va], pva)
        if score > best:  # early stopping on validation PR-AUC
            best, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def main():
    df = load_clean()
    n = len(df)
    i_tr, i_va = int(n * 0.55), int(n * 0.70)
    tf, cf, gf = pd.read_pickle(OUT / "features.pkl")
    gcache = OUT / "granger_tensor.npy"
    if gcache.exists():
        G = np.load(gcache)
    else:
        print("Computing causal graphs for every time step ...")
        G = granger_tensor(df)
        np.save(gcache, G)

    y = future_label(df["fault"].to_numpy())
    normal = df["fault"].to_numpy() == 0
    idx = np.arange(n)
    tr, va, te = idx[:i_tr][normal[:i_tr]], idx[i_tr:i_va][normal[i_tr:i_va]], idx[i_va:][normal[i_va:]]

    per_joint, glob = node_feature_columns(list(tf.columns))
    stacked = np.concatenate([tf.iloc[tr][c].to_numpy(np.float32) for c in per_joint])  # shared per-node scaling
    mu, sd = stacked.mean(0), stacked.std(0) + 1e-6
    gmu, gsd = tf.iloc[tr][glob].to_numpy(np.float32).mean(0), tf.iloc[tr][glob].to_numpy(np.float32).std(0) + 1e-6
    data = lambda rows: tensors(tf, G, rows, per_joint, glob, mu, sd, gmu, gsd)
    print(f"nodes={N_JOINTS}, node features={len(per_joint[0])}, global features={len(glob)}")

    rows = []
    att_example = None
    for name, causal in [("GAT (no causal prior)", False), ("Causal-GAT (ours)", True)]:
        ps, ths = [], []
        for seed in SEEDS:
            model = train_one(causal, seed, data, y, tr, va)
            with torch.no_grad():
                pva = torch.sigmoid(model(*data(va))[0]).numpy()
                xte, gte, Gte = data(te)
                logit, att = model(xte, gte, Gte)
                ps.append(torch.sigmoid(logit).numpy())
            ths.append(best_threshold(y[va], pva))
            if causal and seed == SEEDS[0]:
                att_example = att.numpy()
                beta = [float(model.g1.beta), float(model.g2.beta)]
        p, th = np.mean(ps, 0), float(np.mean(ths))  # seed ensemble
        r = evaluate(y[te], p, th)
        alarm = np.zeros(n - i_va, int)
        alarm[te - i_va] = (p >= th).astype(int)
        r.update(episode_metrics(df["fault"].to_numpy()[i_va:], alarm))
        r.update({"features": "Temporal (per-joint nodes) + causal graph" if causal else "Temporal (per-joint nodes)",
                  "model": name, "n_features": len(per_joint[0]) * N_JOINTS + len(glob)})
        rows.append(r)
        print(f"  {name:24s} F1={r['f1']:.3f} ROC-AUC={r['roc_auc']:.3f} PR-AUC={r['pr_auc']:.3f}")
    print("learned causal-prior weights beta (layer1, layer2):", beta)

    res = pd.read_csv(OUT / "results.csv")
    res = res[~res.model.str.contains("GAT")]
    res = pd.concat([res, pd.DataFrame(rows)], ignore_index=True)
    res.to_csv(OUT / "results.csv", index=False)
    summary = json.loads((OUT / "results.json").read_text())
    summary["results"] = [r for r in summary["results"] if "GAT" not in r["model"]] + rows
    summary["causal_gat_beta"] = beta
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=float))

    # Average attention over test windows that are close to a fault
    near = y[te] == 1
    A = att_example[near].mean(0)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(A, cmap="viridis")
    ax.set(xticks=range(N_JOINTS), yticks=range(N_JOINTS), xticklabels=[f"J{j}" for j in range(N_JOINTS)],
           yticklabels=[f"J{j}" for j in range(N_JOINTS)], xlabel="attends to joint", ylabel="joint",
           title="Causal-GAT attention before faults")
    fig.colorbar(im)
    fig.tight_layout(); fig.savefig(FIG / "gat_attention.png", dpi=150); plt.close(fig)
    print("\n", res[["features", "model", "f1", "roc_auc", "pr_auc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
