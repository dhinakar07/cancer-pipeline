"""
app.py — Streamlit dashboard for the cancer-pipeline project.

Multi-page layout:
  • Overview     — dataset stats, label distribution, pipeline run history
  • Predictions  — per-record predictions, confidence scores, filters
  • Performance  — accuracy over time, confusion matrix
  • Live Predict — type clinical text and get instant Bi-LSTM classification

Demo Mode:
  When no PostgreSQL database is configured (e.g. on Streamlit Cloud),
  the dashboard automatically falls back to realistic sample data so
  recruiters can explore all features without needing a live database.

Run locally:
    streamlit run dashboard/app.py
"""

import os
import sys
import random
from pathlib import Path
from datetime import datetime, timedelta

# ── Add project root to path so src.* imports work ──────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Inject Streamlit Cloud secrets into environment variables ─
# st.secrets are not automatically env vars — this bridges the gap
# so src/config.py (which uses os.getenv) can read them on the cloud.
try:
    for _key, _val in st.secrets.items():
        if _key not in os.environ:
            os.environ[_key] = str(_val)
except Exception:
    pass  # No secrets — will try .env file or fall back to demo mode

# ── Page configuration (must be first Streamlit call) ────────
st.set_page_config(
    page_title="Cancer Classification Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ── Label map — matches training labels ──────────────────────
LABEL_MAP = {0: "Thyroid Cancer", 1: "Colon Cancer", 2: "Lung Cancer"}

# ════════════════════════════════════════════════════════════
# DATABASE CONNECTION — with graceful demo-mode fallback
# ════════════════════════════════════════════════════════════

def _check_db_available() -> bool:
    """
    Return True if a PostgreSQL connection can be established.
    Used once at startup to decide live vs demo mode.
    """
    try:
        from src.db.connection import test_connection
        return test_connection()
    except Exception:
        return False

@st.cache_data(ttl=300)
def query_db(sql: str) -> pd.DataFrame:
    """Run a SELECT query and return a DataFrame (cached 5 min)."""
    from sqlalchemy import text
    from src.db.connection import get_engine
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

# Detect mode once per session
DB_AVAILABLE = _check_db_available()

# ════════════════════════════════════════════════════════════
# DEMO DATA GENERATORS
# All functions return DataFrames that match the exact schema
# of the live database — so the chart/table code is identical
# for both live and demo modes.
# ════════════════════════════════════════════════════════════

@st.cache_data
def _demo_overview_metrics():
    """Return KPI numbers matching a realistic pipeline run."""
    return {
        "total_records": 7500,
        "processed":     7500,
        "total_runs":    5,
        "latest_acc":    99.87,
    }

@st.cache_data
def _demo_label_dist() -> pd.DataFrame:
    """Equal distribution — 2,500 records per cancer type."""
    return pd.DataFrame({
        "label_name": ["Thyroid Cancer", "Colon Cancer", "Lung Cancer"],
        "count":      [2500, 2500, 2500],
    })

@st.cache_data
def _demo_pipeline_runs() -> pd.DataFrame:
    """5 simulated daily pipeline runs with improving accuracy."""
    random.seed(42)
    base = datetime(2026, 5, 23, 2, 0, 0)
    rows = []
    accuracies = [0.9934, 0.9961, 0.9978, 0.9985, 0.9987]
    for i, acc in enumerate(accuracies):
        start = base + timedelta(days=i)
        rows.append({
            "run_id":             f"run_{2026052300 + i}",
            "status":             "success",
            "records_ingested":   7500,
            "records_processed":  7500,
            "records_predicted":  7500,
            "accuracy_pct":       round(acc * 100, 2),
            "started_at":         start.strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at":        (start + timedelta(minutes=random.randint(4, 8))).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return pd.DataFrame(rows)

@st.cache_data
def _demo_predictions() -> pd.DataFrame:
    """
    150 sample prediction rows — realistic confidence scores,
    99.8% correct predictions, matching the live schema.
    """
    random.seed(99)
    cancer_types = ["Thyroid Cancer", "Colon Cancer", "Lung Cancer"]
    sample_texts = {
        "Thyroid Cancer": [
            "Patient presented with neck swelling. Ultrasound revealed a 2.4 cm hypoechoic nodule. Biopsy confirmed papillary thyroid carcinoma staged T2N0M0.",
            "A 45-year-old female referred for evaluation of thyroid nodule. FNA cytology revealed follicular neoplasm. Total thyroidectomy performed.",
            "History of anterior neck discomfort. CT neck demonstrated a 3.0 cm thyroid mass with calcifications. Histopathology showed medullary thyroid carcinoma.",
        ],
        "Colon Cancer": [
            "A 65-year-old male presented with rectal bleeding. Colonoscopy revealed a 3.5 cm circumferential mass at the sigmoid colon. Biopsy confirmed adenocarcinoma.",
            "Patient with iron deficiency anaemia. CT colonography demonstrated a polypoid lesion at the ascending colon. Right hemicolectomy performed with FOLFOX chemotherapy.",
            "A 58-year-old female with change in bowel habits. Biopsy confirmed KRAS-mutated colorectal adenocarcinoma staged T3N1M0.",
        ],
        "Lung Cancer": [
            "A 58-year-old with 30-pack-year smoking history. CT chest showed spiculated 2.7 cm nodule in right upper lobe. Biopsy confirmed non-small cell lung carcinoma.",
            "Patient with persistent haemoptysis. PET-CT demonstrated hypermetabolic activity with mediastinal involvement. Molecular testing confirmed EGFR exon 19 deletion.",
            "A 62-year-old non-smoker with dyspnoea. High-resolution CT revealed ground glass opacity. ALK rearrangement identified. Commenced crizotinib therapy.",
        ],
    }

    rows = []
    for i in range(150):
        true_label  = cancer_types[i % 3]
        # Introduce ~2 errors out of 150 for realism
        is_correct  = (i not in [37, 112])
        pred_label  = true_label if is_correct else cancer_types[(cancer_types.index(true_label) + 1) % 3]

        # Confidence scores — high for correct, lower for errors
        if is_correct:
            top_conf = round(random.uniform(96.5, 99.9), 2)
            rem      = 100 - top_conf
            c1, c2   = round(rem * random.uniform(0.3, 0.7), 2), None
            c2       = round(rem - c1, 2)
        else:
            top_conf = round(random.uniform(52.0, 68.0), 2)
            rem      = 100 - top_conf
            c1       = round(rem * random.uniform(0.3, 0.7), 2)
            c2       = round(rem - c1, 2)

        confs = [top_conf, c1, c2]
        random.shuffle(confs)

        rows.append({
            "original_text":   random.choice(sample_texts[true_label]),
            "true_label":      true_label,
            "predicted_name":  pred_label,
            "is_correct":      is_correct,
            "thyroid_pct":     confs[0],
            "colon_pct":       confs[1],
            "lung_pct":        confs[2],
        })
    return pd.DataFrame(rows)

@st.cache_data
def _demo_accuracy_trend() -> pd.DataFrame:
    """Accuracy trend over 5 pipeline runs — matches live schema."""
    return pd.DataFrame({
        "run_date":          pd.date_range("2026-05-23", periods=5, freq="D"),
        "accuracy_pct":      [99.34, 99.61, 99.78, 99.85, 99.87],
        "records_predicted": [7500, 7500, 7500, 7500, 7500],
    })

@st.cache_data
def _demo_confusion_matrix() -> pd.DataFrame:
    """Near-perfect confusion matrix for 3 cancer classes."""
    labels = ["Thyroid Cancer", "Colon Cancer", "Lung Cancer"]
    data   = [
        ["Thyroid Cancer", "Thyroid Cancer", 2498],
        ["Thyroid Cancer", "Colon Cancer",      1],
        ["Thyroid Cancer", "Lung Cancer",        1],
        ["Colon Cancer",   "Colon Cancer",    2499],
        ["Colon Cancer",   "Thyroid Cancer",     1],
        ["Colon Cancer",   "Lung Cancer",         0],
        ["Lung Cancer",    "Lung Cancer",      2499],
        ["Lung Cancer",    "Thyroid Cancer",      0],
        ["Lung Cancer",    "Colon Cancer",        1],
    ]
    return pd.DataFrame(data, columns=["actual", "predicted", "count"])

# ════════════════════════════════════════════════════════════
# LIVE PREDICT — model inference helper
# ════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_and_tokenizer():
    """
    Load the Bi-LSTM model and tokenizer once and cache them.
    Only called on the Live Predict page when TF is available.
    """
    import pickle
    import tensorflow as tf
    from src.config import MODEL_PATH, TOKENIZER_PATH
    model = tf.keras.models.load_model(str(MODEL_PATH))
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def predict_single(text_input: str) -> dict:
    """
    Run the Bi-LSTM on a single clinical text string.
    Returns predicted label, cancer name, confidence, and all class probs.
    """
    import re
    import nltk
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from src.config import MAX_SEQ_LEN

    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))

    # Preprocess — must match training pipeline exactly
    text_input = text_input.lower()
    text_input = re.sub(r"[^a-z\s]", " ", text_input)
    text_input = re.sub(r"\s+", " ", text_input).strip()
    tokens = [w for w in text_input.split() if w not in stop_words]
    clean  = " ".join(tokens)

    model, tokenizer = load_model_and_tokenizer()
    seq    = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="pre", truncating="pre")
    probs  = model.predict(padded, verbose=0)[0]
    pred   = int(np.argmax(probs))

    return {
        "predicted_label": pred,
        "predicted_name":  LABEL_MAP[pred],
        "confidence":      float(probs[pred]) * 100,
        "all_probs": {
            LABEL_MAP[0]: round(float(probs[0]) * 100, 2),
            LABEL_MAP[1]: round(float(probs[1]) * 100, 2),
            LABEL_MAP[2]: round(float(probs[2]) * 100, 2),
        },
    }

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

st.sidebar.title("🔬 Cancer Pipeline")
st.sidebar.caption("IEEE Bi-LSTM · 99.8% Accuracy")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    options=["Overview", "Predictions", "Performance", "🧪 Live Predict"],
    label_visibility="collapsed",
)

# Demo mode banner — shown on every page when DB is not connected
if not DB_AVAILABLE:
    st.info(
        "📊 **Demo Mode** — showing representative sample data. "
        "Connect a PostgreSQL database and set credentials in `.env` to see live pipeline results.",
        icon="ℹ️",
    )

# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Dataset & Pipeline Overview")
    st.caption("Clinical text records from the Cancer Classification dataset (7,500+ records)")

    # ── KPI metrics ──────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    if DB_AVAILABLE:
        total_records = query_db("SELECT COUNT(*) AS n FROM cancer.raw_records")["n"][0]
        processed     = query_db("SELECT COUNT(*) AS n FROM cancer.processed_records")["n"][0]
        total_runs    = query_db("SELECT COUNT(*) AS n FROM cancer.pipeline_runs")["n"][0]
        latest_acc_df = query_db(
            "SELECT accuracy FROM cancer.pipeline_runs WHERE status='success' "
            "ORDER BY finished_at DESC LIMIT 1"
        )
        latest_acc = float(latest_acc_df["accuracy"][0]) * 100 if not latest_acc_df.empty else 0.0
    else:
        m = _demo_overview_metrics()
        total_records = m["total_records"]
        processed     = m["processed"]
        total_runs    = m["total_runs"]
        latest_acc    = m["latest_acc"]

    col1.metric("Total Records",     f"{total_records:,}")
    col2.metric("Processed Records", f"{processed:,}")
    col3.metric("Pipeline Runs",     f"{total_runs:,}")
    col4.metric("Latest Accuracy",   f"{latest_acc:.2f}%")

    st.markdown("---")

    # ── Label distribution ───────────────────────────────────
    st.subheader("Label Distribution")
    dist_df = (
        query_db("SELECT label_name, COUNT(*) AS count FROM cancer.raw_records GROUP BY label_name")
        if DB_AVAILABLE else _demo_label_dist()
    )
    if not dist_df.empty:
        fig = px.pie(
            dist_df,
            values="count",
            names="label_name",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Recent pipeline runs ─────────────────────────────────
    st.subheader("Recent Pipeline Runs")
    if DB_AVAILABLE:
        runs_df = query_db(
            """
            SELECT run_id, status,
                   records_ingested, records_processed, records_predicted,
                   ROUND(accuracy * 100, 2) AS accuracy_pct,
                   started_at, finished_at
            FROM cancer.pipeline_runs
            ORDER BY started_at DESC LIMIT 10
            """
        )
    else:
        runs_df = _demo_pipeline_runs()

    if runs_df.empty:
        st.info("No pipeline runs recorded yet. Run the Airflow DAG to populate this table.")
    else:
        st.dataframe(runs_df, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTIONS
# ════════════════════════════════════════════════════════════
elif page == "Predictions":
    st.title("Record-Level Predictions")
    st.caption("Bi-LSTM softmax outputs per clinical text record")

    if DB_AVAILABLE:
        run_ids = query_db(
            "SELECT DISTINCT pipeline_run_id FROM cancer.predictions ORDER BY pipeline_run_id DESC"
        )["pipeline_run_id"].tolist()

        if not run_ids:
            st.info("No predictions yet. Trigger the Airflow DAG to generate predictions.")
            st.stop()

        selected_run = st.selectbox("Select Pipeline Run", options=run_ids)
        cancer_filter = st.multiselect(
            "Filter by Predicted Cancer Type",
            options=list(LABEL_MAP.values()),
            default=list(LABEL_MAP.values()),
        )
        pred_df = query_db(
            f"""
            SELECT rr.original_text, rr.label_name AS true_label,
                   p.predicted_name, p.is_correct,
                   ROUND(p.confidence_thyroid * 100, 2) AS thyroid_pct,
                   ROUND(p.confidence_colon   * 100, 2) AS colon_pct,
                   ROUND(p.confidence_lung    * 100, 2) AS lung_pct
            FROM cancer.predictions p
            JOIN cancer.processed_records pr ON pr.id = p.processed_record_id
            JOIN cancer.raw_records rr        ON rr.id = pr.raw_record_id
            WHERE p.pipeline_run_id = '{selected_run}'
            ORDER BY p.id
            """
        )
        pred_df = pred_df[pred_df["predicted_name"].isin(cancer_filter)]
    else:
        # Demo mode — show sample run selector and pre-built predictions
        runs_df     = _demo_pipeline_runs()
        selected_run = st.selectbox("Select Pipeline Run", options=runs_df["run_id"].tolist())
        cancer_filter = st.multiselect(
            "Filter by Predicted Cancer Type",
            options=list(LABEL_MAP.values()),
            default=list(LABEL_MAP.values()),
        )
        pred_df = _demo_predictions()
        pred_df = pred_df[pred_df["predicted_name"].isin(cancer_filter)]

    # ── Accuracy summary ─────────────────────────────────────
    correct = pred_df["is_correct"].sum()
    total   = len(pred_df)
    st.metric(
        label=f"Accuracy for {str(selected_run)[:30]}",
        value=f"{correct / total * 100:.2f}%" if total else "N/A",
        delta=f"{correct:,} / {total:,} correct",
    )

    st.dataframe(pred_df, use_container_width=True, height=400)


# ════════════════════════════════════════════════════════════
# PAGE 3 — PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "Performance":
    st.title("Model Performance Over Time")

    if DB_AVAILABLE:
        acc_df = query_db(
            """
            SELECT finished_at::date AS run_date,
                   ROUND(accuracy * 100, 2) AS accuracy_pct,
                   records_predicted
            FROM cancer.pipeline_runs
            WHERE status = 'success' AND accuracy IS NOT NULL
            ORDER BY finished_at
            """
        )
    else:
        acc_df = _demo_accuracy_trend()

    if acc_df.empty:
        st.info("No completed pipeline runs with accuracy data yet.")
    else:
        st.subheader("Accuracy Over Pipeline Runs")
        fig_acc = px.line(
            acc_df,
            x="run_date",
            y="accuracy_pct",
            markers=True,
            labels={"run_date": "Date", "accuracy_pct": "Accuracy (%)"},
            color_discrete_sequence=["#2ecc71"],
        )
        fig_acc.update_yaxes(range=[98, 101])
        st.plotly_chart(fig_acc, use_container_width=True)

    # ── Confusion matrix ─────────────────────────────────────
    st.subheader("Confusion Matrix — Latest Run")

    if DB_AVAILABLE:
        cm_df = query_db(
            """
            SELECT rr.label_name AS actual, p.predicted_name AS predicted, COUNT(*) AS count
            FROM cancer.predictions p
            JOIN cancer.processed_records pr ON pr.id = p.processed_record_id
            JOIN cancer.raw_records rr        ON rr.id = pr.raw_record_id
            WHERE p.pipeline_run_id = (
                SELECT pipeline_run_id FROM cancer.predictions
                ORDER BY predicted_at DESC LIMIT 1
            )
            GROUP BY rr.label_name, p.predicted_name
            ORDER BY rr.label_name, p.predicted_name
            """
        )
    else:
        cm_df = _demo_confusion_matrix()

    if not cm_df.empty:
        cm_pivot = cm_df.pivot(index="actual", columns="predicted", values="count").fillna(0)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_pivot.values,
            x=cm_pivot.columns.tolist(),
            y=cm_pivot.index.tolist(),
            colorscale="Blues",
            text=cm_pivot.values,
            texttemplate="%{text:.0f}",
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            margin=dict(t=30),
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("No prediction data available.")


# ════════════════════════════════════════════════════════════
# PAGE 4 — LIVE PREDICT
# ════════════════════════════════════════════════════════════
elif page == "🧪 Live Predict":
    st.title("🧪 Live Cancer Classification")
    st.caption("Paste any clinical text and the Bi-LSTM model will classify it instantly")

    # Check if TensorFlow is available
    try:
        import tensorflow as tf
        _tensorflow_available = True
    except ImportError:
        _tensorflow_available = False

    if not _tensorflow_available:
        st.warning(
            "⚠️ **Live Predict requires TensorFlow**, which is not available on this "
            "cloud deployment (Python 3.14 not yet supported by TensorFlow).\n\n"
            "**To use Live Predict, run the dashboard locally:**\n"
            "```bash\nstreamlit run dashboard/app.py\n```\n\n"
            "All other pages work fully in demo mode."
        )
        st.stop()

    st.markdown("---")
    st.subheader("Try a sample or type your own")
    sample = st.selectbox("Load a sample record:", [
        "— type your own below —",
        "Patient presented with neck swelling. Ultrasound revealed a 2.4 cm hypoechoic nodule in the left lobe. Biopsy confirmed papillary thyroid carcinoma staged T2N0M0. Total thyroidectomy was performed.",
        "A 65-year-old male presented with rectal bleeding and change in bowel habits. Colonoscopy revealed a 3.5 cm circumferential mass at the sigmoid colon. Biopsy confirmed moderately differentiated adenocarcinoma.",
        "A 58-year-old female with a 30-pack-year smoking history presented with persistent cough and haemoptysis. CT chest showed a spiculated 2.7 cm nodule in the right upper lobe. Biopsy confirmed non-small cell lung carcinoma.",
    ])

    default_text = "" if sample == "— type your own below —" else sample
    user_text = st.text_area(
        "Enter clinical text here:",
        value=default_text,
        height=180,
        placeholder="e.g. Patient presented with neck swelling. Ultrasound revealed a thyroid nodule...",
    )

    if st.button("🔍 Classify", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some clinical text first.")
        else:
            with st.spinner("Running Bi-LSTM model..."):
                result = predict_single(user_text)

            st.markdown("---")

            cancer_colours = {
                "Thyroid Cancer": "#3498db",
                "Colon Cancer":   "#2ecc71",
                "Lung Cancer":    "#e74c3c",
            }
            colour = cancer_colours.get(result["predicted_name"], "#95a5a6")

            st.markdown(
                f"""
                <div style='background-color:{colour}22; border-left:6px solid {colour};
                            padding:20px; border-radius:8px; margin-bottom:20px'>
                    <h2 style='color:{colour}; margin:0'>{result["predicted_name"]}</h2>
                    <p style='margin:5px 0 0 0; font-size:18px'>
                        Confidence: <b>{result["confidence"]:.2f}%</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("Confidence Scores")
            probs_df = pd.DataFrame({
                "Cancer Type":    list(result["all_probs"].keys()),
                "Confidence (%)": list(result["all_probs"].values()),
            }).sort_values("Confidence (%)", ascending=True)

            fig = px.bar(
                probs_df,
                x="Confidence (%)",
                y="Cancer Type",
                orientation="h",
                color="Cancer Type",
                color_discrete_map={
                    "Thyroid Cancer": "#3498db",
                    "Colon Cancer":   "#2ecc71",
                    "Lung Cancer":    "#e74c3c",
                },
                text="Confidence (%)",
            )
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig.update_layout(showlegend=False, xaxis_range=[0, 105], margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📄 Input text used"):
                st.write(user_text)
