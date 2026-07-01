"""Streamlit dashboard for the Titanic survival classifier.

Run:  streamlit run ds_app.py

Tab 1 — Held-out evaluation: metrics + interactive plots on the test set saved
        by train.py (assignment: "evaluate on a held-out test set in Streamlit").
Tab 2 — Batch inference: give a CSV path (or upload), the trained model is
        loaded from disk, inference runs, and results are shown.

Charts are interactive (Plotly). All model logic lives in evaluate.py.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

import evaluate
from visuals import viz_plotly as viz, plots

ART = "artifacts"

st.set_page_config(page_title="Titanic Survival Model", page_icon="🚢",
                   layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }

.hero { padding: 24px 28px; border-radius: 18px; margin-bottom: 18px;
        background: linear-gradient(135deg, #0F3D3A 0%, #2A9D8F 100%); color: #fff;
        box-shadow: 0 8px 24px rgba(15,61,58,.18); }
.hero-title { font-size: 27px; font-weight: 700; letter-spacing: -.015em; }
.hero-sub { font-size: 14px; opacity: .88; margin-top: 3px; font-weight: 400; }

.kpi { background: #fff; border: 1px solid #E9ECF1; border-radius: 14px;
       padding: 16px 18px; box-shadow: 0 1px 2px rgba(16,24,40,.05); }
.kpi.accent { background: linear-gradient(135deg, #2A9D8F 0%, #21867B 100%); border: none;
              box-shadow: 0 6px 16px rgba(42,157,143,.28); }
.kpi.accent .kpi-val, .kpi.accent .kpi-lab { color: #fff; }
.kpi-val { font-size: 27px; font-weight: 700; color: #1F2A37; line-height: 1.1; }
.kpi-lab { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
           color: #6B7280; margin-top: 4px; font-weight: 600; }

.card { background:#fff; border:1px solid #E9ECF1; border-radius:16px; padding:8px 14px 14px;
        box-shadow:0 1px 2px rgba(16,24,40,.04); }
.sec { font-size:12px; font-weight:700; color:#6B7280; text-transform:uppercase;
       letter-spacing:.06em; margin:14px 0 6px; }
.model-card { background:#F5F7FA; border:1px solid #E9ECF1; border-radius:12px; padding:14px 16px; }
.model-card code { background:#fff; padding:1px 6px; border-radius:6px; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def kpi_row(metrics, accent="roc_auc"):
    cols = st.columns(len(metrics))
    for col, (k, v) in zip(cols, metrics.items()):
        cls = "kpi accent" if k == accent else "kpi"
        col.markdown(
            f'<div class="{cls}"><div class="kpi-val">{v:.3f}</div>'
            f'<div class="kpi-lab">{k.replace("_", " ")}</div></div>',
            unsafe_allow_html=True)


def section(label):
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)


@st.cache_resource
def load_model_cached():
    return evaluate.load_model(ART)


# --------------------------------------------------------------------------- #
# Guard + model
# --------------------------------------------------------------------------- #
if not os.path.exists(os.path.join(ART, "titanic_model.pth")):
    st.error("No trained model found in `artifacts/`. Run `python train.py` first.")
    st.stop()

model, cfg = load_model_cached()
THRESHOLD = float(cfg.get("threshold", 0.5))   # decision threshold used at training time

st.markdown(
    '<div class="hero"><div class="hero-title">🚢 Titanic Survival Model</div>'
    '<div class="hero-sub">PyTorch neural classifier · held-out evaluation & batch inference</div></div>',
    unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sec">Model</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="model-card">'
        f'Input features&nbsp; <code>{cfg.get("input_dim","?")}</code><br>'
        f'Hidden layers&nbsp; <code>{cfg.get("hidden_dims","?")}</code><br>'
        f'Dropout&nbsp; <code>{cfg.get("dropout_rate","?")}</code><br>'
        f'Seed&nbsp; <code>{cfg.get("seed","?")}</code><br>'
        f'Threshold&nbsp; <code>{THRESHOLD:.2f}</code>'
        f'</div>', unsafe_allow_html=True)
    st.caption("Loaded from `artifacts/` (produced by train.py).")

tab_eval, tab_infer = st.tabs(["  Held-out evaluation  ", "  Batch inference  "])


# --------------------------------------------------------------------------- #
# Tab 1 — held-out evaluation
# --------------------------------------------------------------------------- #
with tab_eval:
    X = np.load(os.path.join(ART, "X_test.npy"))
    y = np.load(os.path.join(ART, "y_test.npy"))
    prob = evaluate.predict_proba(model, X)
    metrics = plots.compute_metrics(y, prob, THRESHOLD)

    section(f"Performance on {len(y)} held-out passengers")
    kpi_row(metrics)

    c1, c2 = st.columns([1, 1.25])
    with c1:
        section("Confusion matrix")
        st.plotly_chart(viz.confusion_fig(y, prob, THRESHOLD), use_container_width=True)
    with c2:
        section("ROC & Precision-Recall")
        st.plotly_chart(viz.roc_pr_fig(y, prob), use_container_width=True)

    section("How confidently the model separates the classes")
    st.plotly_chart(viz.prob_dist_fig(y, prob, THRESHOLD), use_container_width=True)

    hist_path = os.path.join(ART, "history.csv")
    if os.path.exists(hist_path):
        with st.expander("Training history"):
            st.plotly_chart(viz.history_fig(pd.read_csv(hist_path).to_dict("list")),
                            use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 2 — batch inference
# --------------------------------------------------------------------------- #
with tab_infer:
    section("Run the trained model on a CSV")
    with st.container(border=True):
        source = st.radio("Data source", ["File path", "Upload CSV"], horizontal=True)
        csv_path = None
        if source == "File path":
            csv_path = st.text_input("Path to a CSV file", value="artifacts/test.csv")
        else:
            up = st.file_uploader("Upload a CSV", type="csv")
            if up is not None:
                csv_path = os.path.join(tempfile.gettempdir(), up.name)
                with open(csv_path, "wb") as f:
                    f.write(up.getbuffer())
        run = st.button("Run inference", type="primary", use_container_width=True)

    if run:
        if not csv_path:
            st.warning("Please provide a CSV path or upload a file.")
        else:
            try:
                res = evaluate.evaluate_csv(csv_path, ART, THRESHOLD)
            except (FileNotFoundError, ValueError) as err:
                st.error(str(err))
            else:
                n_pos = int(res["predictions"].sum())
                section("Summary")
                kpi_row({"records": res["n"], "predicted_survivors": n_pos,
                         "survival_rate": n_pos / res["n"]}, accent="survival_rate")

                if res.get("has_labels"):
                    y_true = res["table"]["Survived"].to_numpy()
                    p = res["probabilities"]
                    section("Evaluation (ground-truth labels found)")
                    kpi_row(res["metrics"])
                    c1, c2 = st.columns([1, 1.25])
                    with c1:
                        st.plotly_chart(viz.confusion_fig(y_true, p, THRESHOLD),
                                        use_container_width=True)
                    with c2:
                        st.plotly_chart(viz.roc_pr_fig(y_true, p), use_container_width=True)
                else:
                    st.info("No `Survived` column — showing predictions only.")

                section("Predictions")
                st.dataframe(res["table"], use_container_width=True, height=280)
                st.download_button("Download predictions (CSV)",
                                   res["table"].to_csv(index=False).encode("utf-8"),
                                   file_name="titanic_predictions.csv", mime="text/csv",
                                   use_container_width=True)