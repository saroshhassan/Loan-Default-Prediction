# System Architecture: Loan Default Risk Assessment

## Overview

This is a production-grade machine learning system designed to predict loan approval risk and support lending decisions. Unlike traditional ML projects, this system bridges the gap between prediction and actionable business decisions.

**Core Objective:** Predict which loan applicants are at risk of default, generate calibrated risk scores, apply decision policies (approve/review/reject), and evaluate the financial impact on portfolio quality.

---

## System Data Flow

```
┌─────────────────┐
│  Raw Dataset    │
│  (Kaggle CSV)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  1. DATA INGESTION              │
│  • Load CSV                     │
│  • Validate schema              │
│  • Check missing values         │
│  • Detect duplicates            │
│  → Output: Validated DataFrame  │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  2. DATA SPLITTING              │
│  • Stratified split (70/15/15)  │
│  • Preserve fraud/approval rate │
│  → Output: train/val/test CSV   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  3. EDA ANALYSIS                │
│  • Feature distributions        │
│  • Approved vs rejected compar. │
│  • Signal strength (IV, ROC)    │
│  • Correlation analysis         │
│  → Output: eda_report.html      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  4. FEATURE ENGINEERING         │
│  • Encode categoricals          │
│  • Scale numericals             │
│  • Create interaction features  │
│  • Feature selection            │
│  → Output: Feature transformer  │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  5. MODELING & TUNING           │
│  • Baseline: Logistic Regression│
│  • Advanced: XGBoost/LightGBM   │
│  • Handle class imbalance       │
│  • Hyperparameter tuning        │
│  → Output: Trained models       │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  6. MODEL CALIBRATION           │
│  • Calibration curve            │
│  • Apply Platt scaling if needed│
│  • SHAP explainability          │
│  → Output: Calibrated model     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  7. DECISIONING ENGINE          │
│  • Risk score → Decision        │
│  • Thresholds: approve/reject   │
│  • Simulate business impact     │
│  → Output: Decision policy      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  8. BUSINESS METRICS            │
│  • Approval rate                │
│  • Default rate in approved     │
│  • Expected loss                │
│  • Tradeoff analysis            │
│  → Output: business_metrics.json│
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  9. INFERENCE INTERFACE         │
│  • API (FastAPI) or Dashboard   │
│  • Real-time scoring            │
│  • SHAP explanations            │
│  → Output: Deployed service     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  10. MONITORING & LOGGING       │
│  • Decision audit trail         │
│  • Model drift detection        │
│  • Portfolio quality tracking   │
└─────────────────────────────────┘
```

---

## Key Components

### 1. Data Ingestion Module (`src/data_ingestion/`)
Loads loan applicant data, validates schema, handles missing values, creates train/val/test splits.

**Key Artifacts:**
- `loader.py` — Load CSV/Parquet
- `validator.py` — Schema validation
- `splitter.py` — Stratified split (preserve approval rate across splits)

**Output:** `data/processed/train.csv`, `val.csv`, `test.csv`

---

### 2. EDA Module (`src/eda/`)
Analyzes feature distributions, approval patterns, signal strength, and multicollinearity.

**Key Artifacts:**
- `distributions.py` — Histogram/KDE plots
- `approval_analysis.py` — Approved vs rejected comparison
- `correlation_analysis.py` — VIF, correlation matrix
- `signal_strength.py` — Information Value (IV), ROC-AUC per feature
- `report_generator.py` — Aggregates into HTML report

**Output:** `reports/eda_report.html`

---

### 3. Feature Engineering Module (`src/feature_engineering/`)
Creates derived features (age groups, income buckets, credit score tiers), encodes categoricals, handles outliers.

**Key Artifacts:**
- `binning.py` — Age/income/credit score bins
- `encoding.py` — Target encoding for categorical features
- `scaling.py` — StandardScaler for numerical features
- `outlier_detection.py` — Flag extreme values
- `selection.py` — Feature importance-based selection
- `pipeline.py` — Orchestrates all feature engineering

**Output:** Feature transformer pickle files in `models/`

---

### 4. Modeling Module (`src/modeling/`)
Trains baseline and advanced models, handles class imbalance, performs hyperparameter tuning.

**Key Artifacts:**
- `models.py` — Model wrappers (Logistic Regression, XGBoost, LightGBM)
- `imbalance_handling.py` — SMOTE, class weights, focal loss
- `tuning.py` — Optuna-based hyperparameter optimization
- `training.py` — Training orchestrator

**Output:** Trained models in `models/best_model.pkl`, `models/base_model.pkl`

---

### 5. Evaluation Module (`src/evaluation/`)
Computes comprehensive metrics (ROC-AUC, PR-AUC, Calibration), generates SHAP explanations.

**Key Artifacts:**
- `metrics.py` — ROC-AUC, PR-AUC, Precision, Recall, F1
- `calibration.py` — Calibration curve, ECE, Platt scaling
- `lift_analysis.py` — Decile analysis, lift chart
- `shap_analysis.py` — SHAP TreeExplainer, feature importance
- `report_generator.py` — Aggregates into JSON

**Output:** `reports/model_report.json`, SHAP plots

---

### 6. Decisioning Module (`src/decisioning/`)
Converts risk scores to decisions (approve/review/reject), simulates business impact.

**Key Artifacts:**
- `policy.py` — Decision class and mapping logic
- `threshold_optimization.py` — Sweep thresholds, compute metrics
- `business_simulator.py` — Simulate approval rate, default rate, expected loss
- `tradeoff_analysis.py` — Generate tradeoff curves

**Output:** `reports/business_metrics.json`, decision policy

---

### 7. Inference Module (`src/inference/`)
Exposes scoring via API or Dashboard.

**Key Artifacts:**
- `model_loader.py` — Load all artifacts (model, calibrator, transformer, explainer)
- `explainer.py` — Generate SHAP explanations

**Output:** Inference service (API or Streamlit)

---

### 8. Utils Module (`src/utils/`)
Shared utilities: logging, config loading, common helpers.

**Key Artifacts:**
- `config_loader.py` — Load YAML configs
- `logging.py` — Structured logging
- `common.py` — Helper functions

---

## Configuration Files

All system behavior is controlled via YAML configs in `configs/`:

- **config.yaml** — Paths, logging, random seed, data paths
- **feature_config.yaml** — Feature definitions, binning ranges, encoding strategies
- **model_config.yaml** — Model types, hyperparameters, imbalance handling
- **decision_config.yaml** — Approval/reject thresholds, business rules

---

## Decision Thresholds & Business Outcomes

The system maps predicted probability of **default** to decisions:

```
Risk Score (P(Default))    Decision      Business Impact
├─ [0.00 - approval_th]   → APPROVE     Low risk, proceed
├─ [approval_th - reject_th] → REVIEW   Medium risk, manual review
└─ [reject_th - 1.00]     → REJECT      High risk, decline

Business Metrics per Threshold:
├─ Approval Rate = % approved
├─ Default Rate (approved) = % of approved that default
├─ Expected Loss = sum(loan_amount × P(default)) for approved
└─ Lift = Default rate / baseline default rate
```

---

## Reproducibility & Versioning

- Fixed random seed in `config.yaml` (default: 42)
- Full config saved with each model run
- Model artifacts timestamped in `models/`
- All outputs logged with timestamp

---

## Error Handling & Logging

- All modules log to `logs/` with module name, timestamp, level (INFO/WARNING/ERROR)
- Failed data validation raises informative errors
- Model training logs validation metrics every N iterations
- Decision policy logs decisions + reasoning for audit trail

---

## Extensibility

The system is designed for easy extension:

1. **Add new features** → Edit `feature_config.yaml` + `src/feature_engineering/`
2. **Try new models** → Edit `model_config.yaml` + add to `src/modeling/models.py`
3. **Change decision logic** → Edit `decision_config.yaml` + `src/decisioning/policy.py`
4. **Add new metrics** → Extend `src/evaluation/metrics.py`

All changes are config-driven; minimal code changes required.
