"""evaluate.py — load the trained Titanic model from disk and evaluate it.

Importable by the Streamlit app (ds_app.py), and runnable standalone:

    python evaluate.py                       # the held-out test set saved by train.py
    python evaluate.py --csv data/test.csv   # any CSV (metrics if it has 'Survived')

Everything is loaded from the artifacts/ folder produced by train.py:
    titanic_model.pth  (weights)   config.json  (how to rebuild the model)
    prep_dict.pkl      (preprocessor)   X_test.npy / y_test.npy (held-out set)
"""

import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd
import torch

from train import TitanicNet            # model class (import does NOT trigger training)
from preprocessing import transform     # apply the fitted preprocessor to new data
from visuals import plots

ARTIFACTS_DIR = "artifacts"

# raw columns the preprocessing pipeline needs to exist in any input CSV
REQUIRED_COLS = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]


# --------------------------------------------------------------------------- #
# Load model / preprocessor from disk
# --------------------------------------------------------------------------- #
def load_model(artifacts_dir=ARTIFACTS_DIR):
    """Rebuild the model from config.json and load the saved weights."""
    cfg_path = os.path.join(artifacts_dir, "config.json")
    w_path = os.path.join(artifacts_dir, "titanic_model.pth")
    if not (os.path.exists(cfg_path) and os.path.exists(w_path)):
        raise FileNotFoundError(
            f"Model files not found in '{artifacts_dir}'. Run train.py first.")
    cfg = json.load(open(cfg_path))
    model = TitanicNet(cfg["input_dim"], cfg["hidden_dims"], cfg["dropout_rate"])
    model.load_state_dict(torch.load(w_path, map_location="cpu"))
    model.eval()
    return model, cfg


def load_preprocessor(artifacts_dir=ARTIFACTS_DIR):
    path = os.path.join(artifacts_dir, "prep_dict.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor not found at '{path}'. Run train.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Prediction + result packaging
# --------------------------------------------------------------------------- #
def predict_proba(model, X):
    """Return survival probabilities for an already-preprocessed feature matrix."""
    X = np.asarray(X, dtype=np.float32)
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).numpy().ravel()


def _results_with_labels(y_true, prob, threshold=0.5):
    """Bundle metrics + figures when ground-truth labels are available."""
    y_true = np.asarray(y_true).astype(int)
    return {
        "has_labels": True,
        "n": len(y_true),
        "probabilities": prob,
        "predictions": (prob >= threshold).astype(int),
        "metrics": plots.compute_metrics(y_true, prob, threshold),
        "fig_confusion": plots.plot_confusion_matrices({"Test": (y_true, prob)}, threshold),
        "fig_roc_pr": plots.plot_roc_pr(y_true, prob),
    }


def evaluate_test_set(artifacts_dir=ARTIFACTS_DIR, threshold=0.5):
    """Evaluate the held-out test set saved by train.py (already preprocessed)."""
    model, _ = load_model(artifacts_dir)
    X = np.load(os.path.join(artifacts_dir, "X_test.npy"))
    y = np.load(os.path.join(artifacts_dir, "y_test.npy"))
    return _results_with_labels(y, predict_proba(model, X), threshold)


def evaluate_csv(csv_path, artifacts_dir=ARTIFACTS_DIR, threshold=0.5):
    """Run inference on a raw CSV.

    If the CSV has a 'Survived' column -> full metrics + plots.
    Otherwise -> inference only (predictions table).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    model, _ = load_model(artifacts_dir)
    prep = load_preprocessor(artifacts_dir)

    X = transform(df, prep)                      # same engineering + fitted transforms
    prob = predict_proba(model, X)
    pred = (prob >= threshold).astype(int)

    table = df.copy()
    table["pred_survived"] = pred
    table["prob_survived"] = prob.round(4)

    if "Survived" in df.columns:
        res = _results_with_labels(df["Survived"], prob, threshold)
        res["table"] = table
        return res
    return {"has_labels": False, "n": len(df),
            "probabilities": prob, "predictions": pred, "table": table}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Evaluate the trained Titanic model.")
    ap.add_argument("--csv", default=None, help="CSV to evaluate (default: saved test set).")
    ap.add_argument("--artifacts", default=ARTIFACTS_DIR)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    res = (evaluate_csv(args.csv, args.artifacts, args.threshold) if args.csv
           else evaluate_test_set(args.artifacts, args.threshold))

    if res["has_labels"]:
        print(f"Evaluated {res['n']} rows:")
        for k, v in res["metrics"].items():
            print(f"  {k:10s} {v:.3f}")
        res["fig_confusion"].savefig(os.path.join(args.artifacts, "screenshots/eval_confusion.png"),
                                     dpi=120, bbox_inches="tight")
        res["fig_roc_pr"].savefig(os.path.join(args.artifacts, "screenshots/eval_roc_pr.png"),
                                  dpi=120, bbox_inches="tight")
        print(f"Figures saved to {args.artifacts}/")
    else:
        print(f"No 'Survived' column -> inference only on {res['n']} rows:")
        print(res["table"][["pred_survived", "prob_survived"]].head().to_string())


if __name__ == "__main__":
    main()