"""Interactive Plotly charts for the Titanic dashboard (hover / zoom)."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

TEAL, CORAL, NAVY, BLUE, GRID = "#2A9D8F", "#E76F51", "#1F2A37", "#4C6EF5", "#EEF1F5"
FONT = dict(family="Inter, sans-serif", color=NAVY)
LABELS = ["Died", "Survived"]

def _base(fig, height=380, title=None):
    fig.update_layout(template="plotly_white", height=height, font=FONT,
                      margin=dict(l=10, r=10, t=46 if title else 20, b=10),
                      title=dict(text=title if title else "", x=0.02, font=dict(size=16, color=NAVY)),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig

def confusion_fig(y_true, y_prob, threshold=0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row = cm / cm.sum(1, keepdims=True).clip(min=1)
    names = [["True Negatives", "False Positives"], ["False Negatives", "True Positives"]]
    text = [[f"<b style='font-size:26px'>{cm[i][j]}</b><br>{names[i][j]} · {row[i][j]:.0%}"
             for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=row, x=[f"Predicted {l}" for l in LABELS], y=[f"Actual {l}" for l in LABELS],
        text=text, texttemplate="%{text}", textfont=dict(size=12, color=NAVY),
        colorscale=[[0, "#FFFFFF"], [0.5, "#BFE3DD"], [1, TEAL]], showscale=False,
        xgap=6, ygap=6, customdata=cm,
        hovertemplate="%{y} → %{x}<br>count = %{customdata}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 400)

def roc_pr_fig(y_true, y_prob):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC curve", "Precision–Recall"))
    fpr, tpr, _ = roc_curve(y_true, y_prob); auc = roc_auc_score(y_true, y_prob)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="#C7CDD6"), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", fill="tozeroy",
                             line=dict(color=TEAL, width=3), name=f"AUC {auc:.3f}",
                             fillcolor="rgba(42,157,143,0.12)",
                             hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>"), 1, 1)
    prec, rec, _ = precision_recall_curve(y_true, y_prob); ap = average_precision_score(y_true, y_prob)
    base = float(np.mean(y_true))
    fig.add_trace(go.Scatter(x=[0, 1], y=[base, base], mode="lines",
                             line=dict(dash="dash", color="#C7CDD6"), showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", fill="tozeroy",
                             line=dict(color=CORAL, width=3), name=f"AP {ap:.3f}",
                             fillcolor="rgba(231,111,81,0.12)",
                             hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra></extra>"), 1, 2)
    fig.update_xaxes(title_text="False positive rate", row=1, col=1)
    fig.update_yaxes(title_text="True positive rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    fig.update_layout(legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))
    return _base(fig, 420)

def history_fig(history):
    tl, vl, va = (np.asarray(history[k]) for k in ("train_loss", "val_loss", "val_acc"))
    ep = np.arange(1, len(tl) + 1); best = int(np.argmin(vl)) + 1
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Validation accuracy"))
    fig.add_trace(go.Scatter(x=ep, y=tl, name="train", line=dict(color=BLUE, width=2.5)), 1, 1)
    fig.add_trace(go.Scatter(x=ep, y=vl, name="val", line=dict(color=CORAL, width=2.5)), 1, 1)
    fig.add_vline(x=best, line=dict(dash="dash", color="#C7CDD6"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep, y=va, name="val acc", line=dict(color=TEAL, width=2.5),
                             showlegend=False), 1, 2)
    fig.add_hline(y=0.616, line=dict(dash="dot", color="#B6BDC7"), row=1, col=2)
    fig.update_xaxes(title_text="epoch"); fig.update_yaxes(range=[0.5, 1.0], row=1, col=2)
    fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
    return _base(fig, 380)


def prob_dist_fig(y_true, y_prob, threshold=0.5):
    """How well the model separates the two classes by predicted probability."""
    yt, yp = np.asarray(y_true), np.asarray(y_prob)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=yp[yt == 0], name="Actually died", marker_color=CORAL,
                               opacity=0.65, nbinsx=28))
    fig.add_trace(go.Histogram(x=yp[yt == 1], name="Actually survived", marker_color=TEAL,
                               opacity=0.65, nbinsx=28))
    fig.add_vline(x=threshold, line=dict(dash="dash", color=NAVY, width=1.5),
                  annotation_text=f"threshold {threshold:.2f}", annotation_position="top")
    fig.update_layout(barmode="overlay",
                      legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))
    fig.update_xaxes(title_text="Predicted survival probability", range=[0, 1])
    fig.update_yaxes(title_text="Passengers")
    return _base(fig, 360)