"""Interactive Hyperparameter Tuning Dashboard for Titanic.

Run: streamlit run interactive_tune.py
"""

import os
import copy
import random
import itertools

import numpy as np
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from preprocessing import preprocess

# Global Plot Styling
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#dee2e6',
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

st.set_page_config(page_title="Titanic Tuning UI", page_icon="🎛️", layout="wide")

# =====================================================================
# BASELINE CONFIGURATION
# =====================================================================
BASELINE_CONFIG = {
    "seed": 42,
    "batch_size": 32,
    "learning_rate": 0.005,
    "epochs": 200,
    "patience": 15,
    "hidden_dims": [64, 32, 16],
    "dropout_rate": 0.2,
}

# Auto-Tune Search Space (Kept small to ensure it runs in < 30 seconds)
AUTO_TUNE_GRID = {
    "learning_rate": [0.001, 0.005, 0.01],
    "batch_size": [16, 32],
    "dropout_rate": [0.2, 0.4],
    "hidden_dims": [[64, 32, 16], [32, 16]]
}

# =====================================================================
# MODEL & UTILS (from train.py)
# =====================================================================
class TitanicNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class EarlyStopping:
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
# CACHED DATA LOADING
# =====================================================================
@st.cache_data
def load_and_preprocess_data(seed):
    """Loads and preprocesses data only once to speed up tuning iterations."""
    path = kagglehub.competition_download("titanic")
    df = pd.read_csv(f"{path}/train.csv")
    X_tr, X_va, X_te, y_tr, y_va, y_te, prep = preprocess(df, seed=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te

# =====================================================================
# TRAINING ENGINE
# =====================================================================
def run_training(config, data_splits):
    """Executes the PyTorch training loop and returns history & best metrics."""
    set_seed(config["seed"])
    X_tr, X_va, _, y_tr, y_va, _ = data_splits

    X_train_t = torch.tensor(X_tr, dtype=torch.float32)
    y_train_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_va, dtype=torch.float32)
    y_val_t = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    model = TitanicNet(
        input_dim=X_tr.shape[1],
        hidden_dims=config["hidden_dims"],
        dropout_rate=config["dropout_rate"],
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    early_stopping = EarlyStopping(patience=config["patience"])

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(config["epochs"]):
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

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_pred = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_pred == y_val_t).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = early_stopping(val_loss)
        if improved:
            best_val_acc = val_acc

        if early_stopping.early_stop:
            break

    return history, best_val_acc, len(history["val_loss"])


def run_auto_tune_grid(data_splits):
    """Generates all grid combinations and finds the best performer."""
    keys = AUTO_TUNE_GRID.keys()
    combinations = list(itertools.product(*AUTO_TUNE_GRID.values()))

    best_acc = 0.0
    best_result = None

    progress_text = "Running Auto-Tune Grid Search..."
    my_bar = st.progress(0, text=progress_text)

    for i, combo in enumerate(combinations):
        # Build config for this run
        current_config = dict(zip(keys, combo))
        # Keep static parameters fixed
        current_config["seed"] = BASELINE_CONFIG["seed"]
        current_config["epochs"] = BASELINE_CONFIG["epochs"]
        current_config["patience"] = BASELINE_CONFIG["patience"]

        my_bar.progress((i + 1) / len(combinations), text=f"Testing configuration {i+1} of {len(combinations)}...")

        c_hist, c_acc, c_epochs = run_training(current_config, data_splits)

        if c_acc > best_acc:
            best_acc = c_acc
            best_result = {
                "history": c_hist,
                "acc": c_acc,
                "epochs": c_epochs,
                "config": current_config
            }

    my_bar.empty()
    return best_result


# =====================================================================
# STREAMLIT UI
# =====================================================================
st.title("🎛️ Titanic Model Fine-Tuning")
st.markdown("Compare custom hyperparameters against the baseline configuration in real-time.")

# Fetch data
data_splits = load_and_preprocess_data(BASELINE_CONFIG["seed"])

# Calculate Baseline (Cached in session state so it only runs once)
if "baseline_results" not in st.session_state:
    with st.spinner("Training baseline model..."):
        b_hist, b_acc, b_epochs = run_training(BASELINE_CONFIG, data_splits)
        st.session_state.baseline_results = {
            "history": b_hist, "acc": b_acc, "epochs": b_epochs
        }

b_res = st.session_state.baseline_results

with st.sidebar:
    st.header("Manual Tuning")

    lr = st.number_input("Learning Rate", value=BASELINE_CONFIG["learning_rate"], format="%.4f", step=0.001)
    bs = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128], value=BASELINE_CONFIG["batch_size"])
    dropout = st.slider("Dropout Rate", 0.0, 0.8, BASELINE_CONFIG["dropout_rate"], 0.05)

    # Parse hidden dims from string
    hidden_dims_str = st.text_input("Hidden Layers (comma separated)", value=", ".join(map(str, BASELINE_CONFIG["hidden_dims"])))
    try:
        hidden_dims = [int(x.strip()) for x in hidden_dims_str.split(",")]
    except ValueError:
        st.error("Invalid format for Hidden Layers. Use comma-separated integers.")
        st.stop()

    patience = st.number_input("Early Stopping Patience", value=BASELINE_CONFIG["patience"], step=5)

    run_btn = st.button("🚀 Train Custom Model", type="primary", use_container_width=True)

    st.divider()
    st.header("🪄 Auto-Tune")
    st.caption("Automatically tests various combinations of LR, Batch Size, Dropout, and Network Depth to find the optimal setup.")
    auto_tune_btn = st.button("Run Auto-Tune Grid Search", type="secondary", use_container_width=True)

# Main Content Area
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Baseline Model")
    st.caption("Standard `train.py` configuration.")
    st.metric("Best Validation Accuracy", f"{b_res['acc']:.2%}")
    st.write(f"**Epochs run:** {b_res['epochs']}")
    st.markdown(f"""
    * **Learning Rate:** `{BASELINE_CONFIG['learning_rate']}`
    * **Batch Size:** `{BASELINE_CONFIG['batch_size']}`
    * **Dropout:** `{BASELINE_CONFIG['dropout_rate']}`
    * **Hidden Dims:** `{BASELINE_CONFIG['hidden_dims']}`
    """)

with col2:
    if not run_btn and not auto_tune_btn and "custom_results" not in st.session_state:
        st.markdown("### Tuned Model")
        st.info("Adjust parameters manually or run Auto-Tune on the left.")

    # Handle Manual Train
    if run_btn:
        custom_config = {
            "seed": BASELINE_CONFIG["seed"], "epochs": BASELINE_CONFIG["epochs"],
            "batch_size": bs, "learning_rate": lr, "dropout_rate": dropout,
            "hidden_dims": hidden_dims, "patience": patience
        }
        with st.spinner("Training custom model..."):
            c_hist, c_acc, c_epochs = run_training(custom_config, data_splits)
            st.session_state.custom_results = {
                "history": c_hist, "acc": c_acc, "epochs": c_epochs, "config": custom_config, "mode": "Manual"
            }

    # Handle Auto Tune
    if auto_tune_btn:
        best_result = run_auto_tune_grid(data_splits)
        best_result["mode"] = "Auto-Tune"
        st.session_state.custom_results = best_result

    # Display Results
    if "custom_results" in st.session_state:
        c_res = st.session_state.custom_results
        cfg = c_res["config"]
        mode = c_res["mode"]

        if mode == "Auto-Tune":
            st.markdown("### 🪄 Auto-Tuned Model")
            st.success("Grid search complete! Found a better (or equal) configuration.")
        else:
            st.markdown("### 🛠️ Custom Tuned Model")

        delta = c_res['acc'] - b_res['acc']
        st.metric("Best Validation Accuracy", f"{c_res['acc']:.2%}", delta=f"{delta:.2%}")
        st.write(f"**Epochs run:** {c_res['epochs']}")

        # Explicitly highlight differences
        st.markdown("**Configuration Changes:**")
        changes_made = False
        for k in ["learning_rate", "batch_size", "dropout_rate", "hidden_dims"]:
            old_v = BASELINE_CONFIG[k]
            new_v = cfg[k]
            if str(old_v) != str(new_v):
                st.markdown(f"* **{k}:** `{old_v}` ➡️ `{new_v}`")
                changes_made = True

        if not changes_made:
            st.markdown("*No changes from baseline.*")

# Plotting
if "custom_results" in st.session_state:
    st.divider()
    st.subheader("Learning Curve Comparison")

    fig, ax = plt.subplots(figsize=(12, 5))

    # Baseline Curve
    b_acc_curve = b_res["history"]["val_acc"]
    ax.plot(range(1, len(b_acc_curve)+1), b_acc_curve, label="Baseline", color="#d62728", linewidth=2.5, alpha=0.8)
    ax.scatter(len(b_acc_curve), b_acc_curve[-1], color="#d62728", s=80, zorder=5)

    # Custom Curve
    c_acc_curve = st.session_state.custom_results["history"]["val_acc"]
    mode_label = st.session_state.custom_results["mode"]
    ax.plot(range(1, len(c_acc_curve)+1), c_acc_curve, label=f"{mode_label} Model", color="#1f77b4", linewidth=2.5, alpha=0.9)
    ax.scatter(len(c_acc_curve), c_acc_curve[-1], color="#1f77b4", s=80, zorder=5)

    ax.set_title("Validation Accuracy Over Epochs", weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.7)

    st.pyplot(fig)