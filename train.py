"""Standalone training script for the Titanic survival classifier.

Run:  python train.py

It does everything end-to-end:
  1. loads the data (from Kaggle),
  2. preprocesses it (leak-free, see preprocessing.py),
  3. trains a small PyTorch MLP with early stopping,
  4. saves the trained weights + preprocessor + metadata to artifacts/.
"""

import os
import json
import copy
import pickle
import random

import numpy as np
import pandas as pd
import kagglehub

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from preprocessing import preprocess


# =====================================================================
# MODEL
# =====================================================================
class TitanicNet(nn.Module):
    """Configurable MLP for tabular binary classification.

    Outputs a single raw logit per row (shape (N, 1)); the sigmoid is applied
    inside BCEWithLogitsLoss during training, so there is NO Sigmoid here.
    """

    def __init__(self, input_dim, hidden_dims=(64, 32, 16), dropout_rate=0.2):
        super().__init__()
        layers = []
        current_dim = input_dim
        # stack: Linear -> ReLU -> Dropout for each hidden size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        # final layer: one logit for binary classification
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# =====================================================================
# UTILITIES
# =====================================================================
def set_seed(seed=42):
    """Lock all random seeds so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Stops training when validation loss stops improving.

    __call__ returns True on the epochs where val loss improved, so the caller
    knows exactly when to snapshot the best weights.
    """

    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        improved = val_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return improved


# =====================================================================
# TRAINING
# =====================================================================
def train():
    # ---- configuration (saved to disk so the model can be rebuilt later) ----
    config = {
        "seed": 42,
        "batch_size": 16,  # optimized
        "learning_rate": 0.005,
        "epochs": 200,
        "patience": 15,
        "hidden_dims": [64, 32, 16],
        "dropout_rate": 0.2,
    }

    set_seed(config["seed"])
    os.makedirs("artifacts", exist_ok=True)

    # ---- 1. load data ----
    print("Downloading dataset...")
    path = kagglehub.competition_download("titanic")
    df = pd.read_csv(f"{path}/train.csv")

    # ---- 2. preprocess (leak-free split + transforms) ----
    print("Preprocessing data...")
    X_tr, X_va, X_te, y_tr, y_va, y_te, prep = preprocess(df, seed=config["seed"])

    # keep the held-out test set for the evaluation/inference step
    np.save("artifacts/X_test.npy", X_te)
    np.save("artifacts/y_test.npy", y_te)

    config["input_dim"] = X_tr.shape[1]

    # ---- tensors & loaders ----
    X_train_t = torch.tensor(X_tr, dtype=torch.float32)
    y_train_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)  # -> (N, 1)
    X_val_t = torch.tensor(X_va, dtype=torch.float32)
    y_val_t = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    # ---- model / loss / optimizer ----
    model = TitanicNet(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        dropout_rate=config["dropout_rate"],
    )
    criterion = nn.BCEWithLogitsLoss()             # sigmoid + BCE in one (stable)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    early_stopping = EarlyStopping(patience=config["patience"])

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_model_state = None

    # ---- 3. training loop ----
    print("Starting training...")
    for epoch in range(config["epochs"]):
        # -- train one epoch --
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)

        # -- validate --
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_pred = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_pred == y_val_t).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # snapshot the weights ONLY when val loss actually improved.
        # copy.deepcopy is essential: state_dict() returns live references, so
        # without it the "best" weights would keep changing as training continues.
        improved = early_stopping(val_loss)
        if improved:
            best_model_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or early_stopping.early_stop:
            print(f"Epoch {epoch + 1:03d}/{config['epochs']} | "
                  f"Train {train_loss:.4f} | Val {val_loss:.4f} | Val Acc {val_acc:.4f}")

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1} "
                  f"(best val loss {early_stopping.best_loss:.4f}).")
            break

    # ---- 4. save artifacts ----
    torch.save(best_model_state, "artifacts/titanic_model.pth")     # best weights
    with open("artifacts/prep_dict.pkl", "wb") as f:                # preprocessor
        pickle.dump(prep, f)
    with open("artifacts/config.json", "w") as f:                   # metadata
        json.dump(config, f, indent=4)
    pd.DataFrame(history).to_csv("artifacts/history.csv", index=False)  # curves

    print("\nDone. Artifacts saved to 'artifacts/'.")


if __name__ == "__main__":
    train()