# 🔬 Cancer Classification Pipeline

> **Portfolio project** by Dhinakar Yalla · Data Engineer  
> Built on an [IEEE-published Bidirectional LSTM](https://ieee.org) achieving **99.8% accuracy** on cancer classification across Thyroid, Colon, and Lung Cancer (7,500+ records).

---

## Overview

An end-to-end data engineering pipeline that ingests clinical text data, preprocesses it with NLP techniques, runs it through a trained Bi-LSTM model, stores results in PostgreSQL, and visualises them in a Streamlit dashboard — all orchestrated by Apache Airflow.

```
Kaggle Dataset
     ↓
[Airflow DAG]
  ├── Task 1: Download CSV (Kaggle API)
  ├── Task 2: Load → cancer.raw_records (PostgreSQL)
  ├── Task 3: NLP Preprocess → cancer.processed_records
  ├── Task 4: Bi-LSTM Predict → cancer.predictions
  └── Task 5: Log run → cancer.pipeline_runs
                              ↓
                    [Streamlit Dashboard]
                    Overview · Predictions · Performance
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Database | PostgreSQL 16 |
| Orchestration | Apache Airflow 2.9 |
| Dashboard | Streamlit + Plotly |
| ML Model | TensorFlow / Keras (Bi-LSTM) |
| Infrastructure | Docker Compose |

## Project Structure

```
cancer-pipeline/
├── data/
│   ├── raw/               ← downloaded Kaggle CSV
│   └── processed/         ← intermediate artefacts
├── src/
│   ├── config.py          ← all env vars in one place
│   ├── db/
│   │   ├── schema.sql     ← PostgreSQL schema (4 tables)
│   │   └── connection.py  ← SQLAlchemy + psycopg2 helpers
│   ├── ingestion/
│   │   ├── kaggle_downloader.py
│   │   └── loader.py      ← CSV → cancer.raw_records
│   ├── transformation/
│   │   └── preprocess.py  ← NLP cleaning + tokenisation
│   └── model/
│       └── predictor.py   ← Bi-LSTM inference + storage
├── dags/
│   └── cancer_pipeline_dag.py   ← Airflow DAG (5 tasks)
├── dashboard/
│   └── app.py             ← Streamlit multi-page app
├── models/                ← place .h5 model + tokenizer.pkl here
├── tests/
│   ├── test_ingestion.py
│   └── test_transformation.py
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/yourusername/cancer-pipeline.git
cd cancer-pipeline
cp .env.example .env
# Edit .env with your PostgreSQL and Kaggle credentials
```

### 2. Start PostgreSQL

```bash
docker-compose up -d
# Schema is applied automatically on first start
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your trained model

Copy your IEEE-published Bi-LSTM model artefacts to:
```
models/bilstm_cancer_classifier.h5
models/tokenizer.pkl
```

### 5. Run the full pipeline manually

```bash
# Download data, preprocess, predict, store results
python -c "
from src.ingestion.kaggle_downloader import download_dataset
from src.ingestion.loader import load_raw_data
from src.transformation.preprocess import preprocess_records
from src.model.predictor import run_predictions
download_dataset()
load_raw_data()
preprocess_records()
run_predictions('manual-run-001')
"
```

### 6. Start the Streamlit dashboard

```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

### 7. Set up Airflow (optional)

```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow scheduler &
airflow webserver --port 8080
# Open http://localhost:8080 and enable the cancer_pipeline DAG
```

## Database Schema

| Table | Description |
|---|---|
| `cancer.raw_records` | Original clinical text + integer labels |
| `cancer.processed_records` | NLP-cleaned text, linked to raw |
| `cancer.predictions` | Bi-LSTM outputs + softmax confidences per run |
| `cancer.pipeline_runs` | Run metadata, accuracy, timestamps |

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Dataset

**Kaggle:** [Clinical Text Classification](https://www.kaggle.com/datasets/ritheshsreenivasan/clinical-text-classification)  
7,500+ clinical medical abstracts labelled across three cancer types:
- **0** — Thyroid Cancer
- **1** — Colon Cancer  
- **2** — Lung Cancer

## Model

Bidirectional LSTM trained and published in IEEE.  
- **Accuracy:** 99.8%  
- **Input:** Padded token sequences (max length 500)  
- **Output:** Softmax probabilities across 3 cancer classes

---

*Built as a data engineering portfolio project to demonstrate production-grade pipeline skills with Python, PostgreSQL, Airflow, and Streamlit.*
