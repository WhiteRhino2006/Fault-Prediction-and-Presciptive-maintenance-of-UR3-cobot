"""Deep time-series baselines (LSTM and GRU) for fault prediction. Offline only - not used in the live demo.

Run after train.py:  python train_lstm.py
Input: the last 30 s of the 19 raw sensor signals (6 joints x current/temp/speed + tool current).
Same label (fault in next 10 s), same chronological split, same normal-state filtering and the same
validation-threshold rule as train.py, so the numbers are directly comparable.
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.data import SENSOR_COLS, load_clean
from src.pipeline import future_label
from train import OUT, best_threshold, episode_metrics, evaluate

SEQ = 30
SEEDS = [0, 1, 2]
EPOCHS = 40


class SeqNet(nn.Module):
    def __init__(self, d_in, cell="lstm", hidden=64):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(d_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


def windows(X, rows):
    """(len(rows), SEQ, d): the SEQ seconds up to and including each row (zero-padded at the start)."""
    pad = np.vstack([np.zeros((SEQ - 1, X.shape[1]), np.float32), X])
    return torch.tensor(np.stack([pad[r:r + SEQ] for r in rows]))


def train_one(cell, seed, Xs, y, tr, va):
    torch.manual_seed(seed)
    xtr, xva = windows(Xs, tr), windows(Xs, va)
    ytr = torch.tensor(y[tr], dtype=torch.float32)
    model = SeqNet(Xs.shape[1], cell)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_w = ((ytr == 0).sum() / max((ytr == 1).sum(), 1)).float()
    best, best_state = -1, None
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(ytr))
        for b in range(0, len(perm), 128):
            i = perm[b:b + 128]
            loss = F.binary_cross_entropy_with_logits(model(xtr[i]), ytr[i], pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            score = average_precision_score(y[va], torch.sigmoid(model(xva)).numpy())
        if score > best:  # early stopping on validation PR-AUC
            best, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def main():
    df = load_clean()
    n = len(df)
    i_tr, i_va = int(n * 0.55), int(n * 0.70)
    X = df[SENSOR_COLS].to_numpy(np.float32)
    mu, sd = X[:i_tr].mean(0), X[:i_tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    y = future_label(df["fault"].to_numpy())
    normal = df["fault"].to_numpy() == 0
    idx = np.arange(n)
    tr, va, te = idx[:i_tr][normal[:i_tr]], idx[i_tr:i_va][normal[i_tr:i_va]], idx[i_va:][normal[i_va:]]

    rows = []
    for cell, name in [("lstm", "LSTM"), ("gru", "GRU")]:
        ps, ths = [], []
        for seed in SEEDS:
            model = train_one(cell, seed, Xs, y, tr, va)
            with torch.no_grad():
                pva = torch.sigmoid(model(windows(Xs, va))).numpy()
                ps.append(torch.sigmoid(model(windows(Xs, te))).numpy())
            ths.append(best_threshold(y[va], pva))
        p, th = np.mean(ps, 0), float(np.mean(ths))  # 3-seed ensemble
        r = evaluate(y[te], p, th)
        alarm = np.zeros(n - i_va, int)
        alarm[te - i_va] = (p >= th).astype(int)
        r.update(episode_metrics(df["fault"].to_numpy()[i_va:], alarm))
        r.update({"features": f"Raw sensor sequence ({SEQ} s)", "model": name, "n_features": len(SENSOR_COLS) * SEQ})
        rows.append(r)
        print(f"  {name:5s} F1={r['f1']:.3f} ROC-AUC={r['roc_auc']:.3f} PR-AUC={r['pr_auc']:.3f}")

    res = pd.read_csv(OUT / "results.csv")
    res = res[~res.model.isin(["LSTM", "GRU"])]
    pd.concat([res, pd.DataFrame(rows)], ignore_index=True).to_csv(OUT / "results.csv", index=False)
    summary = json.loads((OUT / "results.json").read_text())
    summary["results"] = [r for r in summary["results"] if r["model"] not in ("LSTM", "GRU")] + rows
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
