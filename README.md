# 🔬 Cancer Classification Pipeline

> **IEEE-published Bidirectional LSTM achieving 99.8% accuracy** on clinical text classification — operationalised as a production-grade data engineering pipeline with Apache Airflow, AWS (S3 + RDS), and an interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Airflow](https://img.shields.io/badge/Apache%20Airflow-2.9-017CEE?logo=apache-airflow)](https://airflow.apache.org/)
[![AWS](https://img.shields.io/badge/AWS-S3%20%2B%20RDS-FF9900?logo=amazon-aws)](https://aws.amazon.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql)](https://www.postgresql.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)

---

## 📌 What This Project Does

This pipeline ingests 7,500+ clinical text records describing thyroid, colon, and lung cancer cases, preprocesses the text, runs inference through a trained **Bidirectional LSTM** model, stores predictions in a cloud PostgreSQL database, and surfaces results via an interactive dashboard — all orchestrated automatically by Apache Airflow.

| Component | Technology |
|---|---|
| Orchestration | Apache Airflow 2.9 (8-task DAG, @daily schedule) |
| Model | Bi-LSTM (TensorFlow/Keras) — 99.8% accuracy |
| Cloud Storage | AWS S3 (`cancer-pipeline-dhinakar`, us-east-2) |
| Cloud Database | AWS RDS PostgreSQL 16.6 (db.t3.micro) |
| Dashboard | Streamlit — 4 pages including Live Predict |
| NLP | NLTK stop-word removal, Keras Tokenizer, pad_sequences |
| Language | Python 3.11 |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Apache Airflow DAG (@daily)                   │
│                                                                  │
│  download_dataset ──► upload_raw_to_s3 ──► load_raw_data        │
│                                                    │             │
│                                            preprocess_text       │
│                                                    │             │
│                                    upload_processed_to_s3        │
│                                                    │             │
│                                    download_model_from_s3        │
│                                                    │             │
│                                         run_predictions          │
│                                                    │             │
│                                       update_run_metadata        │
└─────────────────────────────────────────────────────────────────┘
          │                                          │
    AWS S3 Bucket                            AWS RDS PostgreSQL
  (raw data + model                        (predictions + run
     artefacts)                               history stored)
          │                                          │
          └──────────── Streamlit Dashboard ─────────┘
                      (Overview · Predictions ·
                       Performance · Live Predict)
```

---

## 🧠 Model — Bidirectional LSTM

Based on my **IEEE-published paper** on cancer classification from clinical text:

```
Embedding(20,000 vocab → 128 dims)
      ↓
Bidirectional LSTM (128 units, return_sequences=True)
      ↓
Dropout (0.3)
      ↓
Bidirectional LSTM (64 units)
      ↓
Dropout (0.3)
      ↓
Dense (64, ReLU)
      ↓
Dropout (0.5)
      ↓
Dense (3, Softmax)  ←  Thyroid / Colon / Lung Cancer
```

**Training results:**
- Test accuracy: **99.8%** (IEEE dataset) / **100%** on synthetic dataset
- Converges in ~4 epochs with early stopping
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

---

## 📁 Project Structure

```
cancer-pipeline/
├── dags/
│   └── cancer_pipeline_dag.py      # 8-task Airflow DAG (cloud-enabled)
├── src/
│   ├── config.py                   # Central config — DB, S3, model paths
│   ├── db/
│   │   ├── connection.py           # SQLAlchemy engine + helpers
│   │   └── schema.sql              # PostgreSQL schema (cancer.* tables)
│   ├── ingestion/
│   │   ├── kaggle_downloader.py    # Kaggle API download (all token formats)
│   │   └── loader.py               # Bulk insert to raw_records table
│   ├── transformation/
│   │   └── preprocess.py           # NLP cleaning + tokenisation
│   ├── model/
│   │   ├── train.py                # Full Bi-LSTM training pipeline
│   │   └── predictor.py            # Batch inference → predictions table
│   └── cloud/
│       └── s3_handler.py           # Upload/download S3 artefacts
├── dashboard/
│   └── app.py                      # Streamlit 4-page dashboard
├── data/
│   └── raw/                        # clinical_text.csv lives here
├── models/                         # bilstm_cancer_classifier.h5 + tokenizer.pkl
├── generate_dataset.py             # Synthetic 7,500-record dataset generator
├── train.sh                        # One-click model training script
├── setup.sh                        # Environment setup script
├── requirements.txt
├── docker-compose.yml              # Local PostgreSQL (optional)
└── .env.example                    # Template — copy to .env, fill credentials
```

---

## ☁️ AWS Infrastructure

| Service | Detail |
|---|---|
| **S3 Bucket** | `cancer-pipeline-dhinakar` (us-east-2) |
| **RDS Instance** | PostgreSQL 16.6 on db.t3.micro |
| **Schema** | `cancer` — 4 tables: raw_records, processed_records, predictions, pipeline_runs |

The pipeline is **cloud-agnostic**: set `S3_BUCKET` in `.env` to enable AWS, or leave it blank to run entirely locally. The Airflow DAG's S3 tasks use `ShortCircuitOperator` to skip gracefully when cloud is not configured.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11
- PostgreSQL (local Docker or AWS RDS)
- Apache Airflow 2.9
- AWS account (optional — for S3 + RDS)

### 1. Clone and set up environment

```bash
git clone https://github.com/dhinakar07/cancer-pipeline.git
cd cancer-pipeline
chmod +x setup.sh && ./setup.sh
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your database credentials and (optionally) AWS keys
```

### 3. Generate dataset and train model

```bash
# Generate 7,500 synthetic clinical records
python generate_dataset.py

# Train the Bi-LSTM (takes 5–15 minutes)
chmod +x train.sh && ./train.sh
```

### 4. Apply database schema

```bash
psql -h <your-host> -U cancer_user -d cancer_pipeline -f src/db/schema.sql
```

### 5. Run the Airflow DAG

```bash
export AIRFLOW_HOME=~/Desktop/cancer-pipeline/airflow
airflow webserver --port 8080 &
airflow scheduler &
# Open http://localhost:8080 → trigger cancer_pipeline DAG
```

### 6. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| **Overview** | KPI metrics, label distribution donut chart, recent pipeline runs |
| **Predictions** | Per-record predictions with confidence scores, filterable by cancer type |
| **Performance** | Accuracy trend over pipeline runs, confusion matrix heatmap |
| **🧪 Live Predict** | Paste any clinical text → instant Bi-LSTM classification with confidence bar chart |

---

## 🗄️ Database Schema

```sql
cancer.raw_records          -- 7,500+ ingested clinical text records
cancer.processed_records    -- cleaned text + token sequences
cancer.predictions          -- model output: label, confidence per class, is_correct
cancer.pipeline_runs        -- run history: status, accuracy, record counts, timestamps
```

---

## 🔬 Cancer Classes

| Label | Cancer Type | Example Finding |
|---|---|---|
| 0 | **Thyroid Cancer** | Papillary thyroid carcinoma, hypoechoic nodule, total thyroidectomy |
| 1 | **Colon Cancer** | Adenocarcinoma, colonoscopy mass, FOLFOX chemotherapy |
| 2 | **Lung Cancer** | NSCLC, spiculated nodule, EGFR mutation, pembrolizumab |

---

## 📄 Research Background

This project operationalises the model from my IEEE paper on clinical text classification using deep learning. The Bidirectional LSTM was chosen because:

- **Bi-directional context**: Medical terms gain meaning from surrounding words in both directions (e.g., "carcinoma" is disambiguated by whether "papillary thyroid" or "non-small cell" precedes it)
- **Sequential modelling**: Clinical text follows narrative structure — symptoms → imaging → pathology → treatment
- **Embedding layer**: Learns domain-specific word representations from the clinical vocabulary

The 99.8% accuracy on 3-class classification demonstrates that NLP-based cancer type identification from clinical abstracts is highly feasible.

---

## 🛠️ Key Engineering Decisions

**Idempotent pipeline** — Each Airflow task checks whether its work is already done before proceeding. Re-running the DAG on the same day is safe.

**Cloud-agnostic design** — A single `cloud_enabled()` flag in `config.py` determines whether AWS tasks run. No code changes needed to switch between local and cloud modes.

**Model caching** — The Streamlit dashboard uses `@st.cache_resource` to load the Bi-LSTM once per session — subsequent Live Predict calls are instantaneous.

**Consistent preprocessing** — `_clean_text()` is defined once and imported by both `train.py` and `predictor.py`, ensuring training and inference use identical text representations.

---

## 📬 Contact

**Dhinakar Yalla** — Data Engineer
📧 dhinakaryalla07@gmail.com
🐙 [github.com/dhinakar07](https://github.com/dhinakar07)

---

*Built to demonstrate end-to-end data engineering skills: cloud infrastructure (AWS S3 + RDS), pipeline orchestration (Apache Airflow), ML inference at scale (TensorFlow Bi-LSTM), and interactive data products (Streamlit).*
