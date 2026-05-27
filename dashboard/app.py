"""
app.py — Streamlit entry point for the cancer-pipeline dashboard.

Multi-page layout:
  • Overview     — dataset stats, label distribution, pipeline run history
  • Predictions  — per-record predictions, confidence scores, filters
  • Performance  — accuracy over time, confusion matrix

Run with:
    streamlit run dashboard/app.py

The dashboard connects to PostgreSQL using the same config as the pipeline.
"""

import os
import sys
from pathlib import Path

# ── Add project root to path so src.* imports work ──────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Inject Streamlit Cloud secrets into environment variables ─
# Streamlit Cloud stores secrets in st.secrets, but src/config.py
# reads credentials via os.getenv(). This block bridges the two
# so the dashboard works on both local (.env file) and cloud (st.secrets).
try:
    for _key, _val in st.secrets.items():
        if _key not in os.environ:
            os.environ[_key] = str(_val)
except Exception:
    pass  # No secrets configured — will fall back to .env file

# ── Page configuration (must be the first Streamlit call) ───
st.set_page_config(
    page_title="Cancer Classification Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ──────────────────────────────────────────────────
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text

from src.config import LABEL_MAP
from src.db.connection import get_engine, test_connection


# ── Database helper with caching ────────────────────────────
@st.cache_data(ttl=300)   # cache for 5 minutes; refresh automatically
def query_db(sql: str) -> pd.DataFrame:
    """
    Execute a SELECT query and return a DataFrame.
    Results are cached for 5 minutes to avoid hammering Postgres.
    """
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


# ── Live prediction helper (no DB needed) ───────────────────
@st.cache_resource
def load_model_and_tokenizer():
    """
    Load the Bi-LSTM model and tokenizer once and cache them.
    Subsequent calls return the cached objects instantly.
    """
    import pickle
    import tensorflow as tf
    from src.config import MODEL_PATH, TOKENIZER_PATH
    model = tf.keras.models.load_model(str(MODEL_PATH))
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def predict_single(text: str) -> dict:
    """
    Run the Bi-LSTM model on a single clinical text input.

    Returns a dict with:
        predicted_label  — 0 / 1 / 2
        predicted_name   — Thyroid / Colon / Lung Cancer
        confidence       — softmax probability of top class (%)
        all_probs        — {label_name: probability} for all 3 classes
    """
    import re
    import nltk
    import numpy as np
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from src.config import MAX_SEQ_LEN

    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))

    # Preprocess text — same steps as training
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in stop_words]
    clean  = " ".join(tokens)

    # Tokenise and pad
    model, tokenizer = load_model_and_tokenizer()
    seq    = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="pre", truncating="pre")

    # Predict
    probs = model.predict(padded, verbose=0)[0]
    pred  = int(np.argmax(probs))

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


# ── Sidebar navigation ───────────────────────────────────────
st.sidebar.title("🔬 Cancer Pipeline")
st.sidebar.caption("IEEE Bi-LSTM · 99.8% Accuracy")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    options=["Overview", "Predictions", "Performance", "🧪 Live Predict"],
    label_visibility="collapsed",
)

# ── Database connectivity check ──────────────────────────────
if not test_connection():
    st.error(
        "⚠️ Cannot connect to PostgreSQL. "
        "Check your `.env` settings and make sure the database is running."
    )
    st.stop()


# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Dataset & Pipeline Overview")
    st.caption("Clinical text records from the Kaggle Cancer Classification dataset (7,500+ records)")

    # ── KPI metrics row ──────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    total_records = query_db("SELECT COUNT(*) AS n FROM cancer.raw_records")["n"][0]
    total_runs    = query_db("SELECT COUNT(*) AS n FROM cancer.pipeline_runs")["n"][0]
    latest_acc    = query_db(
        "SELECT accuracy FROM cancer.pipeline_runs WHERE status='success' "
        "ORDER BY finished_at DESC LIMIT 1"
    )
    latest_acc    = float(latest_acc["accuracy"][0]) * 100 if not latest_acc.empty else 0.0
    processed     = query_db("SELECT COUNT(*) AS n FROM cancer.processed_records")["n"][0]

    col1.metric("Total Records",    f"{total_records:,}")
    col2.metric("Processed Records",f"{processed:,}")
    col3.metric("Pipeline Runs",    f"{total_runs:,}")
    col4.metric("Latest Accuracy",  f"{latest_acc:.2f}%")

    st.markdown("---")

    # ── Label distribution chart ─────────────────────────────
    st.subheader("Label Distribution")
    dist_df = query_db(
        "SELECT label_name, COUNT(*) AS count FROM cancer.raw_records GROUP BY label_name"
    )
    if not dist_df.empty:
        fig = px.pie(
            dist_df,
            values="count",
            names="label_name",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,          # donut chart
        )
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Recent pipeline runs table ───────────────────────────
    st.subheader("Recent Pipeline Runs")
    runs_df = query_db(
        """
        SELECT run_id, status,
               records_ingested, records_processed, records_predicted,
               ROUND(accuracy * 100, 2) AS accuracy_pct,
               started_at, finished_at
        FROM cancer.pipeline_runs
        ORDER BY started_at DESC
        LIMIT 10
        """
    )
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

    # ── Run selector ─────────────────────────────────────────
    run_ids = query_db(
        "SELECT DISTINCT pipeline_run_id FROM cancer.predictions ORDER BY pipeline_run_id DESC"
    )["pipeline_run_id"].tolist()

    if not run_ids:
        st.info("No predictions yet. Trigger the Airflow DAG to generate predictions.")
        st.stop()

    selected_run = st.selectbox("Select Pipeline Run", options=run_ids)

    # ── Filter by cancer type ────────────────────────────────
    cancer_filter = st.multiselect(
        "Filter by Predicted Cancer Type",
        options=list(LABEL_MAP.values()),
        default=list(LABEL_MAP.values()),
    )

    # ── Load predictions for selected run ────────────────────
    pred_df = query_db(
        f"""
        SELECT
            p.id,
            rr.original_text,
            rr.label_name       AS true_label,
            p.predicted_name,
            p.is_correct,
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

    # Apply type filter
    pred_df = pred_df[pred_df["predicted_name"].isin(cancer_filter)]

    # ── Summary row ──────────────────────────────────────────
    correct = pred_df["is_correct"].sum()
    total   = len(pred_df)
    st.metric(
        label=f"Accuracy for {selected_run[:30]}…",
        value=f"{correct / total * 100:.2f}%" if total else "N/A",
        delta=f"{correct:,} / {total:,} correct",
    )

    # ── Prediction table ─────────────────────────────────────
    st.dataframe(
        pred_df.drop(columns=["id"]),
        use_container_width=True,
        height=400,
    )


# ════════════════════════════════════════════════════════════
# PAGE 3 — PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "Performance":
    st.title("Model Performance Over Time")

    # ── Accuracy trend ───────────────────────────────────────
    acc_df = query_db(
        """
        SELECT
            finished_at::date                   AS run_date,
            ROUND(accuracy * 100, 2)            AS accuracy_pct,
            records_predicted
        FROM cancer.pipeline_runs
        WHERE status = 'success' AND accuracy IS NOT NULL
        ORDER BY finished_at
        """
    )

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
        fig_acc.update_yaxes(range=[90, 101])   # zoom into meaningful range
        st.plotly_chart(fig_acc, use_container_width=True)

    # ── Confusion matrix (latest run) ───────────────────────
    st.subheader("Confusion Matrix — Latest Run")
    cm_df = query_db(
        """
        SELECT
            rr.label_name   AS actual,
            p.predicted_name AS predicted,
            COUNT(*)         AS count
        FROM cancer.predictions p
        JOIN cancer.processed_records pr ON pr.id = p.processed_record_id
        JOIN cancer.raw_records rr        ON rr.id = pr.raw_record_id
        WHERE p.pipeline_run_id = (
            SELECT pipeline_run_id
            FROM cancer.predictions
            ORDER BY predicted_at DESC
            LIMIT 1
        )
        GROUP BY rr.label_name, p.predicted_name
        ORDER BY rr.label_name, p.predicted_name
        """
    )

    if not cm_df.empty:
        # Pivot into matrix shape
        cm_pivot = cm_df.pivot(index="actual", columns="predicted", values="count").fillna(0)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_pivot.values,
            x=cm_pivot.columns.tolist(),
            y=cm_pivot.index.tolist(),
            colorscale="Blues",
            text=cm_pivot.values,
            texttemplate="%{text}",
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            margin=dict(t=30),
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("No prediction data available for confusion matrix.")


# ════════════════════════════════════════════════════════════
# PAGE 4 — LIVE PREDICT
# ════════════════════════════════════════════════════════════
elif page == "🧪 Live Predict":
    st.title("🧪 Live Cancer Classification")
    st.caption("Paste any clinical text and the Bi-LSTM model will classify it instantly")

    # Check if TensorFlow is available (not available on Streamlit Cloud with Python 3.14)
    try:
        import tensorflow as tf
        _tensorflow_available = True
    except ImportError:
        _tensorflow_available = False

    if not _tensorflow_available:
        st.warning(
            "⚠️ **Live Predict is not available in this cloud deployment.**\n\n"
            "TensorFlow does not yet support Python 3.14 on Streamlit Cloud. "
            "To use Live Predict, run the dashboard locally:\n\n"
            "```bash\nstreamlit run dashboard/app.py\n```\n\n"
            "All other pages (Overview, Predictions, Performance) work fully in the cloud."
        )
        st.stop()

    st.markdown("---")

    # ── Sample texts for quick testing ──────────────────────
    st.subheader("Try a sample or type your own")
    sample = st.selectbox("Load a sample record:", [
        "— type your own below —",
        "Patient presented with neck swelling. Ultrasound revealed a 2.4 cm hypoechoic nodule in the left lobe. Biopsy confirmed papillary thyroid carcinoma staged T2N0M0. Total thyroidectomy was performed.",
        "A 65-year-old male presented with rectal bleeding and change in bowel habits. Colonoscopy revealed a 3.5 cm circumferential mass at the sigmoid colon. Biopsy confirmed moderately differentiated adenocarcinoma.",
        "A 58-year-old female with a 30-pack-year smoking history presented with persistent cough and haemoptysis. CT chest showed a spiculated 2.7 cm nodule in the right upper lobe. Biopsy confirmed non-small cell lung carcinoma.",
    ])

    # ── Text input ───────────────────────────────────────────
    default_text = "" if sample == "— type your own below —" else sample
    user_text = st.text_area(
        "Enter clinical text here:",
        value=default_text,
        height=180,
        placeholder="e.g. Patient presented with neck swelling. Ultrasound revealed a thyroid nodule...",
    )

    # ── Predict button ───────────────────────────────────────
    if st.button("🔍 Classify", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some clinical text first.")
        else:
            with st.spinner("Running Bi-LSTM model..."):
                result = predict_single(user_text)

            st.markdown("---")

            # ── Result banner ────────────────────────────────
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
                    <h2 style='color:{colour}; margin:0'>
                        {result["predicted_name"]}
                    </h2>
                    <p style='margin:5px 0 0 0; font-size:18px'>
                        Confidence: <b>{result["confidence"]:.2f}%</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Confidence bar chart ─────────────────────────
            st.subheader("Confidence Scores")
            probs_df = pd.DataFrame({
                "Cancer Type": list(result["all_probs"].keys()),
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
            fig.update_layout(
                showlegend=False,
                xaxis_range=[0, 105],
                margin=dict(t=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Input summary ────────────────────────────────
            with st.expander("📄 Input text used"):
                st.write(user_text)
