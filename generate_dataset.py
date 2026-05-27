"""
generate_dataset.py — Generate a realistic synthetic cancer classification dataset.

Creates 7,500 clinical text records (2,500 per class) matching the structure
of the Kaggle Clinical Text Classification dataset:

  Columns:
    medical_abstract  — clinical text description
    condition_label   — 0 (Thyroid), 1 (Colon), 2 (Lung)

Each record is built by randomly combining real medical terminology,
symptoms, procedures, and findings for each cancer type — producing
natural-sounding clinical abstracts suitable for NLP training.

Output: data/raw/clinical_text.csv

Usage:
    python generate_dataset.py
"""

import random
import os
import sys
from pathlib import Path
import pandas as pd
from loguru import logger

# ── Reproducible output ──────────────────────────────────────
random.seed(42)

# ── Output path ──────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
OUT_PATH = ROOT / "data" / "raw" / "clinical_text.csv"

RECORDS_PER_CLASS = 2500   # 2,500 × 3 = 7,500 total

# ════════════════════════════════════════════════════════════
# CLINICAL TEXT VOCABULARY PER CANCER TYPE
# Terms sourced from real medical literature / textbooks.
# ════════════════════════════════════════════════════════════

# ── Label 0: Thyroid Cancer ──────────────────────────────────
THYROID = {
    "demographics": [
        "A {age}-year-old {sex} presented with",
        "Patient is a {age}-year-old {sex} who presented with",
        "A {age}-year-old {sex} was referred for evaluation of",
        "History was obtained from a {age}-year-old {sex} complaining of",
    ],
    "symptoms": [
        "a palpable neck mass", "neck swelling", "dysphagia",
        "hoarseness of voice", "a thyroid nodule", "anterior neck discomfort",
        "difficulty swallowing", "a painless neck lump", "cervical lymphadenopathy",
        "progressive neck enlargement",
    ],
    "imaging": [
        "Ultrasound revealed a hypoechoic nodule measuring {size} cm in the {lobe} lobe",
        "CT neck demonstrated a {size} cm thyroid mass with calcifications",
        "MRI showed a heterogeneous thyroid lesion involving the {lobe} lobe",
        "Ultrasound-guided FNA was performed on a {size} cm solid nodule",
        "Scintigraphy showed a cold nodule in the {lobe} lobe of the thyroid",
    ],
    "pathology": [
        "Biopsy confirmed papillary thyroid carcinoma",
        "Fine needle aspiration cytology revealed follicular neoplasm",
        "Histopathology showed medullary thyroid carcinoma",
        "Pathology confirmed anaplastic thyroid carcinoma",
        "FNAC was consistent with Hurthle cell neoplasm",
        "Frozen section revealed papillary microcarcinoma",
        "Cytology showed follicular carcinoma with capsular invasion",
    ],
    "staging": [
        "staged as T{t}N{n}M{m}", "staged pT{t}N{n}M{m}",
        "AJCC stage {stage}", "classified as stage {stage} disease",
    ],
    "treatment": [
        "Total thyroidectomy was performed with central neck dissection",
        "Patient underwent hemithyroidectomy followed by radioactive iodine therapy",
        "Near-total thyroidectomy was completed uneventfully",
        "Radioiodine ablation was administered post-operatively",
        "TSH suppression therapy was initiated with levothyroxine",
        "Modified radical neck dissection was performed for nodal disease",
        "Patient was started on thyroid hormone replacement therapy",
    ],
    "followup": [
        "Serum thyroglobulin levels were undetectable at follow-up",
        "Whole body scan showed no evidence of residual or metastatic disease",
        "Patient remained disease-free at 12-month follow-up",
        "Repeat ultrasound at 6 months showed no recurrence",
        "TSH-stimulated thyroglobulin was within normal limits",
    ],
}

# ── Label 1: Colon Cancer ────────────────────────────────────
COLON = {
    "demographics": [
        "A {age}-year-old {sex} was admitted with",
        "A {age}-year-old {sex} presented to the gastroenterology clinic with",
        "Patient is a {age}-year-old {sex} referred for colonoscopy after",
        "A {age}-year-old {sex} with a family history of colorectal cancer presented with",
    ],
    "symptoms": [
        "rectal bleeding", "change in bowel habits", "unexplained weight loss",
        "iron deficiency anaemia", "abdominal pain", "constipation and diarrhoea alternating",
        "melena", "hematochezia", "a palpable abdominal mass",
        "fatigue and anaemia", "obstipation", "tenesmus",
    ],
    "imaging": [
        "Colonoscopy revealed a {size} cm circumferential mass at the {location}",
        "CT colonography demonstrated a polypoid lesion at the {location}",
        "Abdominal CT showed a {size} cm colonic mass with pericolonic fat stranding",
        "PET-CT revealed hypermetabolic activity at the {location} with hepatic metastases",
        "Colonoscopy identified an obstructing lesion at the {location}",
    ],
    "pathology": [
        "Biopsy confirmed moderately differentiated adenocarcinoma",
        "Histopathology revealed well-differentiated colorectal adenocarcinoma",
        "Pathology showed poorly differentiated mucinous adenocarcinoma",
        "Biopsy demonstrated signet ring cell carcinoma",
        "Immunohistochemistry confirmed MSI-high colorectal carcinoma",
        "Pathology showed T3 adenocarcinoma with lymphovascular invasion",
        "Biopsy confirmed KRAS-mutated colorectal adenocarcinoma",
    ],
    "staging": [
        "staged as T{t}N{n}M{m}", "Dukes stage {dukes}",
        "TNM stage {stage}", "AJCC stage {stage}",
    ],
    "treatment": [
        "Right hemicolectomy was performed with primary anastomosis",
        "Anterior resection was completed with diverting loop ileostomy",
        "Laparoscopic sigmoid colectomy was performed",
        "FOLFOX chemotherapy was initiated post-operatively",
        "Patient received CAPOX regimen with bevacizumab",
        "Abdominoperineal resection was performed for low rectal involvement",
        "Neoadjuvant chemoradiotherapy was administered prior to surgery",
    ],
    "followup": [
        "CEA levels normalised post-operatively",
        "Follow-up CT at 3 months showed no evidence of recurrence",
        "Patient tolerated chemotherapy well with no grade 3 toxicity",
        "Surveillance colonoscopy at 12 months was unremarkable",
        "Carcinoembryonic antigen levels remained within normal limits",
    ],
}

# ── Label 2: Lung Cancer ─────────────────────────────────────
LUNG = {
    "demographics": [
        "A {age}-year-old {sex} with a {pack}-pack-year smoking history presented with",
        "A {age}-year-old {sex} ex-smoker presented with",
        "Patient is a {age}-year-old {sex} who presented with",
        "A {age}-year-old {sex} non-smoker presented with",
    ],
    "symptoms": [
        "persistent cough", "haemoptysis", "dyspnoea", "chest pain",
        "unintentional weight loss", "recurrent pneumonia", "hoarseness",
        "superior vena cava syndrome", "Pancoast syndrome symptoms",
        "progressive shortness of breath", "finger clubbing", "wheezing",
    ],
    "imaging": [
        "Chest X-ray revealed a {size} cm opacity in the {lobe} lobe",
        "CT chest showed a spiculated {size} cm nodule in the {lobe} lobe",
        "PET-CT demonstrated hypermetabolic activity in the {lobe} lobe with mediastinal involvement",
        "CT guided biopsy of a {size} cm right upper lobe mass was performed",
        "High-resolution CT revealed ground glass opacity with {size} cm solid component",
    ],
    "pathology": [
        "Biopsy confirmed non-small cell lung carcinoma, adenocarcinoma subtype",
        "Histopathology revealed squamous cell carcinoma of the lung",
        "Pathology showed small cell lung carcinoma with extensive disease",
        "Biopsy demonstrated large cell neuroendocrine carcinoma",
        "Molecular testing confirmed EGFR exon 19 deletion adenocarcinoma",
        "ALK rearrangement was identified on FISH testing",
        "PD-L1 expression was 80% on immunohistochemistry",
    ],
    "staging": [
        "staged as T{t}N{n}M{m}", "AJCC stage {stage}",
        "clinical stage {stage}", "staged as limited stage disease",
    ],
    "treatment": [
        "Lobectomy was performed via video-assisted thoracoscopic surgery",
        "Patient commenced erlotinib targeted therapy for EGFR-mutated disease",
        "Platinum-based doublet chemotherapy was initiated",
        "Stereotactic body radiotherapy was delivered to the primary lesion",
        "Pembrolizumab immunotherapy was commenced as first-line treatment",
        "Pneumonectomy was performed for central tumour location",
        "Osimertinib was prescribed for T790M-mutated progressive disease",
    ],
    "followup": [
        "CT at 3 months demonstrated significant tumour response",
        "Patient achieved partial response after 4 cycles of chemotherapy",
        "Disease remained stable on maintenance therapy at 6-month imaging",
        "Repeat bronchoscopy showed no endobronchial recurrence",
        "Brain MRI was negative for metastatic disease",
    ],
}

# ── Shared fill-in values ────────────────────────────────────
AGES   = list(range(35, 85))
SEXES  = ["male", "female"]
SIZES  = ["1.2", "1.8", "2.1", "2.4", "2.7", "3.0", "3.5", "4.1", "4.8", "5.2"]
LOBES  = ["left", "right", "upper", "lower", "middle", "left upper", "right lower"]
STAGES = ["I", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IV", "IVA"]
DUKES  = ["A", "B", "C", "D"]
LOCATIONS = [
    "sigmoid colon", "ascending colon", "descending colon",
    "transverse colon", "rectosigmoid junction", "caecum",
    "hepatic flexure", "splenic flexure",
]
T_STAGES = ["1", "2", "3", "4", "4a"]
N_STAGES = ["0", "1", "2", "2a"]
M_STAGES = ["0", "1", "1a"]
PACKS    = list(range(10, 60, 5))


def _fill(template: str) -> str:
    """Replace all {placeholders} in a template with random values."""
    return template.format(
        age      = random.choice(AGES),
        sex      = random.choice(SEXES),
        size     = random.choice(SIZES),
        lobe     = random.choice(LOBES),
        location = random.choice(LOCATIONS),
        stage    = random.choice(STAGES),
        dukes    = random.choice(DUKES),
        t        = random.choice(T_STAGES),
        n        = random.choice(N_STAGES),
        m        = random.choice(M_STAGES),
        pack     = random.choice(PACKS),
    )


def _generate_record(vocab: dict) -> str:
    """
    Build one clinical abstract by randomly picking and combining
    sentences from each section of the vocabulary.

    Structure:
      demographic intro → symptom → imaging finding →
      pathology result → staging → treatment → follow-up
    """
    parts = [
        _fill(random.choice(vocab["demographics"]))
        + " " + random.choice(vocab["symptoms"]) + ".",

        _fill(random.choice(vocab["imaging"])) + ".",

        random.choice(vocab["pathology"])
        + ", " + _fill(random.choice(vocab["staging"])) + ".",

        random.choice(vocab["treatment"]) + ".",

        random.choice(vocab["followup"]) + ".",
    ]
    # Randomly drop 1–2 sentences for variety
    num_sentences = random.randint(3, 5)
    selected = random.sample(parts, k=num_sentences)
    return " ".join(selected)


def generate(n_per_class: int = RECORDS_PER_CLASS) -> pd.DataFrame:
    """
    Generate n_per_class records for each of the 3 cancer types.

    Returns:
        DataFrame with columns: medical_abstract, condition_label
    """
    records = []

    cancer_types = [
        (THYROID, 0, "Thyroid Cancer"),
        (COLON,   1, "Colon Cancer"),
        (LUNG,    2, "Lung Cancer"),
    ]

    for vocab, label, name in cancer_types:
        logger.info(f"Generating {n_per_class:,} records for {name}...")
        for _ in range(n_per_class):
            text = _generate_record(vocab)
            records.append({"medical_abstract": text, "condition_label": label})

    df = pd.DataFrame(records)

    # Shuffle so the three classes are interleaved
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def main() -> None:
    logger.info("=" * 55)
    logger.info("  cancer-pipeline — Synthetic Dataset Generator")
    logger.info("=" * 55)

    df = generate()

    # Save to data/raw/
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    logger.info(f"Saved {len(df):,} records to {OUT_PATH}")
    logger.info(f"Class distribution:\n{df['condition_label'].value_counts().to_string()}")
    logger.info(f"\nSample record (Thyroid):\n{df[df.condition_label==0].iloc[0].medical_abstract}\n")
    logger.info(f"Sample record (Colon):\n{df[df.condition_label==1].iloc[0].medical_abstract}\n")
    logger.info(f"Sample record (Lung):\n{df[df.condition_label==2].iloc[0].medical_abstract}\n")
    logger.info("✅ Dataset ready at data/raw/clinical_text.csv")


if __name__ == "__main__":
    main()
