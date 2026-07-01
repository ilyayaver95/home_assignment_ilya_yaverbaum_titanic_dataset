"""Plotting + metrics helpers for the training run and test evaluation.

All functions return matplotlib Figures, so they can be saved to disk after
training and rendered in Streamlit (st.pyplot) with no changes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score,
)

DIED, SURVIVED, TRAIN = "#E07A5F", "#3D9A8B", "#5B8DEF"
CM_CMAP = LinearSegmentedColormap.from_list("teal", ["#FFFFFF", SURVIVED])
LABELS = ["Died", "Survived"]


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Standard binary-classification metrics from probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


def plot_training_curves(history):
    """Loss (train vs val) and validation accuracy, with the best epoch marked."""
    tl = np.asarray(history["train_loss"]); vl = np.asarray(history["val_loss"])
    va = np.asarray(history["val_acc"]); epochs = np.arange(1, len(tl) + 1)
    best = int(np.argmin(vl))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # loss: shaded generalization gap + overfit zone after the best epoch
    ax[0].fill_between(epochs, tl, vl, color=DIED, alpha=0.07)
    ax[0].axvspan(best + 1, epochs[-1], color=DIED, alpha=0.04)
    ax[0].plot(epochs, tl, color=TRAIN, lw=2, label="train")
    ax[0].plot(epochs, vl, color=DIED, lw=2, label="validation")
    ax[0].scatter([best + 1], [vl[best]], s=70, color=DIED, edgecolor="white", lw=1.5, zorder=5)
    ax[0].annotate(f"best {vl[best]:.3f}\nepoch {best + 1}", (best + 1, vl[best]),
                   textcoords="offset points", xytext=(12, 8), fontsize=9, color="#555")
    ax[0].set_title("Loss over epochs", fontweight="bold", loc="left")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("BCE loss"); ax[0].legend(frameon=False)
    # accuracy
    ax[1].plot(epochs, va, color=SURVIVED, lw=2, label="val accuracy")
    ax[1].axhline(0.616, ls=":", color="#bbb", label="majority baseline (62%)")
    ax[1].axvline(best + 1, ls="--", color="#999", lw=1)
    ax[1].scatter([best + 1], [va[best]], s=70, color=SURVIVED, edgecolor="white", lw=1.5, zorder=5)
    ax[1].set_title("Validation accuracy over epochs", fontweight="bold", loc="left")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].set_ylim(0.5, 1.0)
    ax[1].legend(frameon=False, loc="lower right")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", color="#EEE")
    fig.tight_layout()
    return fig


def _confusion_panel(ax, y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_row = cm / cm.sum(axis=1, keepdims=True).clip(min=1)   # row-normalized (recall)
    ax.imshow(cm_row, cmap=CM_CMAP, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{cm_row[i, j]:.0%}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if cm_row[i, j] > 0.55 else "#333")
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
    ax.set_title(f"{name}  \u00b7  acc {acc:.0%}", fontweight="bold", fontsize=12)
    ax.set_xticks([0, 1], [f"Pred\n{l}" for l in LABELS], fontsize=9)
    ax.set_yticks([0, 1], [f"True\n{l}" for l in LABELS], fontsize=9)
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", lw=3); ax.tick_params(which="both", length=0)


def plot_confusion_matrices(sets, threshold=0.5):
    """`sets` = {"Train": (y_true, y_prob), "Validation": (...), "Test": (...)}.

    Cells show count and row % (per-true-class recall). Works with 1+ panels.
    """
    n = len(sets)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, (name, (yt, yp)) in zip(axes, sets.items()):
        _confusion_panel(ax, yt, (np.asarray(yp) >= threshold).astype(int), name)
    fig.suptitle("Confusion matrices", fontweight="bold", fontsize=14, x=0.02, ha="left")
    fig.tight_layout()
    return fig


def plot_roc_pr(y_true, y_prob):
    """Threshold-independent test-set curves: ROC and Precision-Recall."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(y_true, y_prob); auc = roc_auc_score(y_true, y_prob)
    ax[0].plot(fpr, tpr, color=SURVIVED, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax[0].plot([0, 1], [0, 1], ls="--", color="#bbb", label="random")
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[0].set_title("ROC curve", fontweight="bold", loc="left"); ax[0].legend(frameon=False, loc="lower right")
    prec, rec, _ = precision_recall_curve(y_true, y_prob); ap = average_precision_score(y_true, y_prob)
    ax[1].plot(rec, prec, color=DIED, lw=2, label=f"PR (AP = {ap:.3f})")
    ax[1].axhline(np.mean(y_true), ls="--", color="#bbb", label=f"baseline ({np.mean(y_true):.2f})")
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
    ax[1].set_title("Precision-Recall curve", fontweight="bold", loc="left"); ax[1].legend(frameon=False, loc="lower left")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(color="#EEE")
    fig.tight_layout()
    return fig