"""Extra prediction baselines requested by the research plan: SVM, 1D-CNN and Transformer (offline only).

Run after train.py:  python train_baselines_extra.py
  SVM          RBF kernel on the same temporal + causal + game features as our model
  1D-CNN       temporal convolutions over the last 30 s of the 19 raw sensor signals
  Transformer  self-attention encoder over the same 30-s sequences
Same label (fault in the next 10 s), chronological split, normal-state filtering and validation-threshold
rule as train.py / train_lstm.py, so all numbers are directly comparable. Kept out of the dashboard.
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import SENSOR_COLS, load_clean
from src.pipeline import future_label
from train import OUT, best_threshold, episode_metrics, evaluate
from train_lstm import SEEDS, SEQ, windows

EPOCHS = 40
NEW = ["SVM (RBF)", "1D-CNN", "Transformer"]


class CNN1D(nn.Module):
    def __init__(self, d_in, ch=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(d_in, ch, 5, padding=2), nn.ReLU(), nn.Conv1d(ch, ch, 5, padding=2),
                                 nn.ReLU(), nn.AdaptiveMaxPool1d(1))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(ch, 1))

    def forward(self, x):  # x: (B, T, d)
        return self.head(self.net(x.transpose(1, 2)).squeeze(-1)).squeeze(-1)


class TinyTransformer(nn.Module):
    def __init__(self, d_in, d_model=32, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.randn(1, SEQ, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=64, dropout=0.2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.enc(self.inp(x) + self.pos)
        return self.head(h[:, -1]).squeeze(-1)


def train_seq(make, seed, Xs, y, tr, va):
    torch.manual_seed(seed)
    xtr, xva = windows(Xs, tr), windows(Xs, va)
    ytr = torch.tensor(y[tr], dtype=torch.float32)
    model = make()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_w = ((ytr == 0).sum() / max((ytr == 1).sum(), 1)).float()
    best, state = -1, None
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(ytr))
        for b in range(0, len(perm), 128):
            i = perm[b:b + 128]
            loss = F.binary_cross_entropy_with_logits(model(xtr[i]), ytr[i], pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s = average_precision_score(y[va], torch.sigmoid(model(xva)).numpy())
        if s > best:
            best, state = s, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model


def report(name, feats, y, te, n, i_va, p, th, df):
    r = evaluate(y[te], p, th)
    alarm = np.zeros(n - i_va, int)
    alarm[te - i_va] = (p >= th).astype(int)
    r.update(episode_metrics(df["fault"].to_numpy()[i_va:], alarm))
    r.update({"features": feats, "model": name})
    print(f"  {name:12s} F1={r['f1']:.3f} ROC-AUC={r['roc_auc']:.3f} PR-AUC={r['pr_auc']:.3f}", flush=True)
    return r


def main():
    df = load_clean()
    n = len(df)
    i_tr, i_va = int(n * 0.55), int(n * 0.70)
    y = future_label(df["fault"].to_numpy())
    normal = df["fault"].to_numpy() == 0
    idx = np.arange(n)
    tr, va, te = idx[:i_tr][normal[:i_tr]], idx[i_tr:i_va][normal[i_tr:i_va]], idx[i_va:][normal[i_va:]]
    rows = []

    # --- SVM on our engineered features (temporal + causal + game) ---
    tf, cf, gf = pd.read_pickle(OUT / "features.pkl")
    X = pd.concat([tf, cf, gf], axis=1)
    X = X.loc[:, ~X.columns.duplicated()].to_numpy(float)
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"))
    svm.fit(X[tr], y[tr])
    th = best_threshold(y[va], svm.decision_function(X[va]))
    svm.fit(X[np.r_[tr, va]], y[np.r_[tr, va]])
    rows.append(report("SVM (RBF)", "Full (T+C+G)", y, te, n, i_va, svm.decision_function(X[te]), th, df))

    # --- sequence models on 30 s of raw sensors ---
    R = df[SENSOR_COLS].to_numpy(np.float32)
    mu, sd = R[:i_tr].mean(0), R[:i_tr].std(0) + 1e-6
    Rs = (R - mu) / sd
    for name, make in [("1D-CNN", lambda: CNN1D(Rs.shape[1])), ("Transformer", lambda: TinyTransformer(Rs.shape[1]))]:
        ps, ths = [], []
        for seed in SEEDS:
            m = train_seq(make, seed, Rs, y, tr, va)
            with torch.no_grad():
                ths.append(best_threshold(y[va], torch.sigmoid(m(windows(Rs, va))).numpy()))
                ps.append(torch.sigmoid(m(windows(Rs, te))).numpy())
        rows.append(report(name, f"Raw sensor sequence ({SEQ} s)", y, te, n, i_va, np.mean(ps, 0),
                           float(np.mean(ths)), df))

    res = pd.read_csv(OUT / "results.csv")
    res = res[~res.model.isin(NEW)]
    pd.concat([res, pd.DataFrame(rows)], ignore_index=True).to_csv(OUT / "results.csv", index=False)
    summary = json.loads((OUT / "results.json").read_text())
    summary["results"] = [r for r in summary["results"] if r["model"] not in NEW] + rows
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
