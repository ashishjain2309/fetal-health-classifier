# =========================================================
# PROJECT A — Fetal Health Classification: Streamlit app
# Loads the five saved pipelines from models/ (no retraining),
# predicts on an uploaded CSV and reports all six metrics.
# =========================================================

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

MODEL_DIR = Path(__file__).parent / "models"

st.set_page_config(page_title="Fetal Health Classifier", page_icon="🩺", layout="wide")


# ---------------------------------------------------------
# LOAD SAVED ARTEFACTS (cached: loaded once, never retrained)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in meta["models"].items()}
    return meta, models


@st.cache_data
def load_saved_metrics():
    return pd.read_csv(MODEL_DIR / "metrics_project_a.csv", index_col=0)


meta, models = load_artifacts()
TARGET = meta["target"]
FEATURES = meta["feature_columns"]
CLASS_NAMES = {int(k): v for k, v in meta["class_names"].items()}
LABELS = sorted(CLASS_NAMES)                      # [1, 2, 3]


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def read_csv_safely(source):
    """Read a CSV, returning (dataframe, error message)."""
    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        return None, "The file is empty."
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        return None, f"The file could not be read as a CSV ({type(e).__name__})."
    df.columns = [str(c).strip() for c in df.columns]          # tolerate stray spaces in headers
    if df.empty:
        return None, "The file has column headers but no data rows."
    return df, None


def check_columns(df):
    """Return the list of required feature columns missing from the uploaded file."""
    return [c for c in FEATURES if c not in df.columns]


def compute_metrics(y_true, y_pred, y_proba):
    """Six metrics, macro-averaged for the three classes; AUC is one-vs-rest."""
    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC (OvR, macro)": np.nan,
        "Precision (macro)": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    # AUC needs every class to appear in the true labels
    if set(LABELS).issubset(set(y_true)):
        results["AUC (OvR, macro)"] = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                                    average="macro", labels=LABELS)
    return results


# ---------------------------------------------------------
# SIDEBAR: model choice and data source
# ---------------------------------------------------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload a test CSV", type="csv")
use_sample = st.sidebar.checkbox("Use the bundled test set instead", value=uploaded is None)

st.sidebar.download_button("Download the bundled test set",
                           data=(MODEL_DIR / "test_set.csv").read_bytes(),
                           file_name="test_set.csv", mime="text/csv")
st.sidebar.caption(f"Models trained with scikit-learn {meta['sklearn_version']} · "
                   f"running {sklearn.__version__}")


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🩺 Fetal Health Classification")
st.write("Classifies cardiotocography (CTG) records as **Normal**, **Suspect** or **Pathological** "
         "using five pre-trained scikit-learn models. Upload a CSV with the 20 CTG features "
         "(and optionally the `fetal_health` label) to get predictions and evaluation metrics.")

tab_eval, tab_compare, tab_about = st.tabs(["Predict & evaluate", "Model comparison", "About the data"])


# ---------------------------------------------------------
# TAB 1: PREDICT & EVALUATE
# ---------------------------------------------------------
with tab_eval:
    if uploaded is not None and not use_sample:
        df, read_error = read_csv_safely(uploaded)
        source = f"uploaded file **{uploaded.name}**"
    elif use_sample:
        df, read_error = read_csv_safely(MODEL_DIR / "test_set.csv")
        source = "the bundled **test_set.csv** (423 held-out records)"
    else:
        st.info("Upload a CSV in the sidebar, or tick **Use the bundled test set**.")
        st.stop()
    if read_error:
        st.error(read_error)
        st.stop()

    missing = check_columns(df)
    if missing:
        st.error(f"The file is missing {len(missing)} required column(s): {missing}")
        st.stop()

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    bad_rows = X.isna().any(axis=1)
    if bad_rows.any():
        st.warning(f"{int(bad_rows.sum())} row(s) contain missing or non-numeric values and were skipped.")
        X, df = X[~bad_rows], df[~bad_rows]
    if X.empty:
        st.error("No usable rows left after removing invalid values.")
        st.stop()

    model = models[model_name]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    st.subheader(f"{model_name} on {len(X)} records")
    st.caption(f"Data source: {source}")

    has_labels = TARGET in df.columns
    if has_labels:
        # Accept numeric labels (1, 2, 3 or 1.0 ...) or text labels (Normal, Suspect, Pathological)
        name_to_code = {v.lower(): k for k, v in CLASS_NAMES.items()}
        raw = df[TARGET]
        as_text = raw.astype(str).str.strip().str.lower().map(name_to_code)
        y_true = pd.to_numeric(raw, errors="coerce").fillna(as_text).round().astype("Int64")
        valid = y_true.isin(LABELS)
        if not valid.all():
            st.warning(f"{int((~valid).sum())} row(s) have a `{TARGET}` value other than 1, 2 or 3; "
                       "they are excluded from the metrics.")
        y_true_v = y_true[valid].astype(int).to_numpy()
        y_pred_v, y_proba_v = y_pred[valid.to_numpy()], y_proba[valid.to_numpy()]

        # Six metrics
        scores = compute_metrics(y_true_v, y_pred_v, y_proba_v)
        cols = st.columns(6)
        for col, (name, value) in zip(cols, scores.items()):
            col.metric(name, "n/a" if pd.isna(value) else f"{value:.4f}")
        if pd.isna(scores["AUC (OvR, macro)"]):
            st.caption("AUC needs all three classes in the uploaded labels, so it is not shown.")
        st.caption("Precision, recall and F1 are macro-averaged: each class counts equally, "
                   "so the rare Pathological class is not hidden by the large Normal class.")

        # Confusion matrix + classification report
        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion matrix**")
            cm = confusion_matrix(y_true_v, y_pred_v, labels=LABELS)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            ConfusionMatrixDisplay(cm, display_labels=[CLASS_NAMES[l] for l in LABELS]).plot(
                ax=ax, cmap="Blues", colorbar=False)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)
        with right:
            st.markdown("**Classification report**")
            report = classification_report(y_true_v, y_pred_v, labels=LABELS,
                                           target_names=[CLASS_NAMES[l] for l in LABELS],
                                           output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")
    else:
        st.info(f"No `{TARGET}` column found, so only predictions are shown (metrics need true labels).")

    # Predictions table + download
    st.markdown("**Predictions**")
    out = df.copy()
    out["predicted_class"] = y_pred
    out["predicted_label"] = [CLASS_NAMES[int(c)] for c in y_pred]
    for i, c in enumerate(model.classes_):
        out[f"prob_{CLASS_NAMES[int(c)]}"] = y_proba[:, i].round(4)
    st.dataframe(out, width="stretch", height=300)
    st.download_button("Download predictions as CSV", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                       mime="text/csv")


# ---------------------------------------------------------
# TAB 2: MODEL COMPARISON (saved results on the held-out test set)
# ---------------------------------------------------------
with tab_compare:
    st.subheader("All five models on the held-out test set")
    saved = load_saved_metrics()
    st.dataframe(saved.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"),
                 width="stretch")
    st.caption("Green = best value in each column. Computed once after training on the "
               "423-record stratified test set; shown here without retraining.")


# ---------------------------------------------------------
# TAB 3: ABOUT THE DATA
# ---------------------------------------------------------
with tab_about:
    st.markdown(f"""
**Dataset:** [Fetal Health Classification (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification),
2,126 CTG records labelled by expert obstetricians.

**Target:** `{TARGET}` — 1 = Normal, 2 = Suspect, 3 = Pathological (about 78% / 14% / 8%).

**Preprocessing:** 13 exact duplicate rows removed; `histogram_width` dropped (it always equals
max − min); stratified 80/20 split with random state {meta['random_state']}. Scaling or power
transforms live inside each saved pipeline and were fitted on the training data only.

**Models:** Logistic Regression, Decision Tree, KNN (k = 5), Gaussian Naive Bayes and an
RBF-kernel SVM with calibrated probabilities. Class weighting is used where the model supports it.

**Required input columns ({len(FEATURES)}):** {", ".join(f"`{c}`" for c in FEATURES)}
""")
