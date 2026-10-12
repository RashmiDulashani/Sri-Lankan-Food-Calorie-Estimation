"""Compute Macro-F1, per-class F1 and macro AUC-ROC from saved predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

THRESHOLD = 0.5


def score(true, probs, names):
    """Per-class F1 and AUC. A class with no positives in this set is skipped (F1 undefined)."""
    pred = (probs >= THRESHOLD).astype(int)
    f1s, aucs = {}, {}
    for j, c in enumerate(names):
        t = true[:, j]
        if t.sum() == 0:
            continue
        f1s[c] = f1_score(t, pred[:, j], zero_division=0)
        if t.sum() < len(t):
            aucs[c] = roc_auc_score(t, probs[:, j])
    return f1s, aucs


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved predictions.")
    parser.add_argument("--predictions", default="results/metrics/resnet50_predictions.csv")
    parser.add_argument("--out-dir", default="results/metrics")
    parser.add_argument("--name", default="resnet50")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    names = [c[5:] for c in df.columns if c.startswith("true_")]

    rows = []
    for fold in sorted(df.fold.unique()):
        part = df[df.fold == fold]
        true = part[[f"true_{c}" for c in names]].values.astype(int)
        probs = part[[f"prob_{c}" for c in names]].values
        f1s, aucs = score(true, probs, names)
        row = {"fold": fold, "photos": len(part),
               "macro_f1": np.mean(list(f1s.values())),
               "macro_auc": np.mean(list(aucs.values())),
               "classes_scored": len(f1s)}
        row.update({f"f1_{c}": f1s.get(c, np.nan) for c in names})
        rows.append(row)
    by_fold = pd.DataFrame(rows)

    print("PER FOLD (classes absent from a fold's validation set are skipped):")
    print(by_fold.round(3).to_string(index=False))

    print()
    print("MEAN +- SD ACROSS FOLDS:")
    for col in ["macro_f1", "macro_auc"]:
        print(f"  {col:10s} {by_fold[col].mean():.3f} +- {by_fold[col].std():.3f}")

    true_all = df[[f"true_{c}" for c in names]].values.astype(int)
    probs_all = df[[f"prob_{c}" for c in names]].values
    f1s, aucs = score(true_all, probs_all, names)
    print()
    print("ALL 46 PHOTOS POOLED (each photo predicted once, by the model that never saw its plate):")
    for c in names:
        print(f"  {c:15s} F1 {f1s[c]:.3f}   AUC {aucs[c]:.3f}")
    print(f"  Macro-F1 {np.mean(list(f1s.values())):.3f}   Macro AUC {np.mean(list(aucs.values())):.3f}")

    out_dir = Path(args.out_dir)
    by_fold.to_csv(out_dir / f"{args.name}_metrics_by_fold.csv", index=False)
    print()
    print("Saved:", out_dir / f"{args.name}_metrics_by_fold.csv")


if __name__ == "__main__":
    main()