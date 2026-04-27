# Module Reference: Purpose & Responsibilities

## Overview

Each module in `src/` has a specific responsibility in the ML pipeline. This document outlines what each module does, its key functions, and its dependencies.

---

## 1. Data Ingestion Module (`src/data_ingestion/`)

**Purpose:** Load raw loan data, validate schema, handle missing values, create train/validation/test splits.

**Responsibilities:**
- Load dataset from CSV/Parquet file
- Validate column names and data types
- Check for missing values and duplicates
- Log dataset statistics (shape, fraud rate, feature distributions)
- Create stratified train/val/test splits (70/15/15)
- Preserve target variable distribution across splits

**Key Functions:**
- `load_dataset(path: str) → pd.DataFrame` — Load data
- `validate_schema(df: pd.DataFrame, config: dict) → bool` — Validate schema
- `check_missing_values(df: pd.DataFrame) → dict` — Report missing %
- `stratified_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) → Tuple[pd.DataFrame, ...]` — Split data
- `log_dataset_stats(df: pd.DataFrame)` — Log stats

**Dependencies:** pandas, numpy

**Output Artifacts:** `data/processed/train.csv`, `val.csv`, `test.csv`

**Config Reference:** `feature_config.yaml` specifies target column and categorical features

---

## 2. EDA Module (`src/eda/`)

**Purpose:** Explore data distributions, identify patterns, analyze signal strength.

**Responsibilities:**
- Generate distribution plots for all numerical features
- Generate bar plots for categorical features
- Compare approved vs rejected applicant distributions
- Compute correlation matrix and VIF (multicollinearity)
- Calculate signal strength metrics (Information Value, ROC-AUC per feature)
- Identify weak signals and strong separators
- Aggregate all visualizations into HTML report

**Key Functions:**
- `plot_distributions(df: pd.DataFrame) → None` — Save distribution plots
- `plot_approval_comparison(df: pd.DataFrame, target: str) → None` — Approved vs rejected
- `compute_correlation_matrix(df: pd.DataFrame) → pd.DataFrame` — Correlation
- `compute_vif(df: pd.DataFrame) → pd.DataFrame` — Multicollinearity
- `compute_information_value(df: pd.DataFrame, target: str) → pd.DataFrame` — IV per feature
- `compute_roc_auc_per_feature(df: pd.DataFrame, target: str) → pd.DataFrame` — ROC-AUC univariate
- `generate_html_report(output_path: str)` — Aggregate report

**Dependencies:** pandas, numpy, matplotlib, seaborn, plotly, scikit-learn

**Output Artifacts:** `reports/eda_report.html`, individual PNG plots

**Config Reference:** `feature_config.yaml` specifies categorical vs numerical features

---

## 3. Feature Engineering Module (`src/feature_engineering/`)

**Purpose:** Transform raw features into ML-ready features with no data leakage.

**Responsibilities:**
- Encode categorical features (target encoding, frequency encoding)
- Scale numerical features (StandardScaler, RobustScaler)
- Create binned features (age groups, income brackets, credit score tiers)
- Handle outliers (flag or cap extreme values)
- Feature selection (importance-based or correlation-based)
- Fit transformers on train set only
- Apply transformers to val/test without refitting (no leakage)

**Key Classes:**
- `FeatureEngineeringPipeline` — Orchestrates all transformations
  - `.fit(X_train, y_train)` — Fit on train only
  - `.transform(X)` → Transformed features
  - `.fit_transform(X_train, y_train)` → Fit + transform on train

**Key Functions:**
- `encode_categoricals(df: pd.DataFrame, strategy: str) → pd.DataFrame` — Target/frequency encoding
- `scale_numericals(df: pd.DataFrame, method: str) → pd.DataFrame` — StandardScaler
- `create_bins(df: pd.DataFrame, columns: dict) → pd.DataFrame` — Binning
- `detect_outliers(df: pd.DataFrame, method: str) → pd.DataFrame` — Flag outliers
- `select_features(df: pd.DataFrame, importances: np.ndarray, k: int) → list` — Top-K features

**Dependencies:** pandas, numpy, scikit-learn, category_encoders

**Output Artifacts:** Fitted transformers saved as pickle in `models/`

**Config Reference:** `feature_config.yaml` defines binning ranges, encoding strategy, scaling method

**Critical:** Prevent data leakage by fitting only on train split

---

## 4. Modeling Module (`src/modeling/`)

**Purpose:** Train baseline and advanced ML models with class imbalance handling and hyperparameter tuning.

**Responsibilities:**
- Implement model wrappers (consistent interface for all models)
- Train baseline model (Logistic Regression) for interpretability
- Train advanced models (XGBoost, LightGBM)
- Handle class imbalance (SMOTE, class weights, sample weighting)
- Perform hyperparameter tuning via Optuna
- Log training metrics and validation performance
- Save trained models and best hyperparameters

**Key Classes:**
- `BaseModel` — Abstract base class
  - `.fit(X, y)` → Train
  - `.predict_proba(X)` → Probability predictions
  - `.save(path: str)` → Serialize model
  - `.load(path: str)` → Deserialize model

- `LogisticRegressionModel(BaseModel)`
- `XGBoostModel(BaseModel)`
- `LightGBMModel(BaseModel)`

**Key Functions:**
- `apply_smote(X_train, y_train) → (X_train_smote, y_train_smote)` — Oversample minority
- `train_baseline_model(X_train, y_train, X_val, y_val) → BaseModel` — Logistic Regression
- `train_advanced_models(X_train, y_train, X_val, y_val, config) → dict` — XGBoost + LightGBM
- `optimize_hyperparameters(X_train, y_train, X_val, y_val, model_type: str) → dict` — Optuna tuning

**Dependencies:** pandas, numpy, scikit-learn, xgboost, lightgbm, optuna, imbalanced-learn

**Output Artifacts:** Trained models in `models/`, best_params.json

**Config Reference:** `model_config.yaml` specifies models to train, hyperparameter ranges, imbalance strategy

---

## 5. Evaluation Module (`src/evaluation/`)

**Purpose:** Comprehensively evaluate model performance beyond accuracy.

**Responsibilities:**
- Compute ROC-AUC and PR-AUC
- Calculate precision, recall, F1 at multiple thresholds
- Generate calibration curves
- Compute expected calibration error (ECE)
- Perform decile analysis (lift chart)
- Calculate KS statistic
- Generate SHAP explanations
- Compile all metrics into JSON report

**Key Functions:**
- `compute_roc_auc(y_true, y_pred_proba) → float` — ROC-AUC
- `compute_pr_auc(y_true, y_pred_proba) → float` — PR-AUC
- `compute_precision_recall_f1(y_true, y_pred, threshold) → dict` — Per-threshold metrics
- `plot_calibration_curve(y_true, y_pred_proba) → calibrator` — Platt scaling if needed
- `apply_calibration(y_pred_proba, calibrator) → y_pred_proba_calibrated` — Calibrate scores
- `compute_lift_chart(y_true, y_pred_proba) → pd.DataFrame` — Decile analysis
- `compute_ks_statistic(y_true, y_pred_proba) → float` — KS stat
- `generate_shap_explanations(model, X_test) → (explainer, shap_values)` — SHAP values
- `generate_evaluation_report(metrics_dict, output_path)` — JSON report

**Dependencies:** pandas, numpy, scikit-learn, shap, matplotlib, seaborn

**Output Artifacts:** `reports/model_report.json`, SHAP plots, calibration plots

---

## 6. Decisioning Module (`src/decisioning/`)

**Purpose:** Convert risk scores to actionable decisions and simulate business impact.

**Responsibilities:**
- Define decision classes (APPROVE, REVIEW, REJECT)
- Map risk scores to decisions based on thresholds
- Simulate business outcomes for different threshold pairs:
  - Approval rate (% of applications approved)
  - Default rate in approved portfolio
  - False negative rate (% of actual defaults not caught)
  - Expected loss (sum of loan amount × P(default) for approved)
- Generate tradeoff curves (approval vs default capture, etc.)
- Perform risk segmentation analysis

**Key Classes:**
- `Decision` (Enum) → APPROVE, REVIEW, REJECT

**Key Functions:**
- `apply_decision_policy(risk_scores, approve_threshold, reject_threshold) → pd.Series[Decision]` — Score to decision
- `simulate_business_impact(decisions, y_true, loan_amounts) → dict` — Compute business metrics
- `optimize_thresholds(y_true, y_pred_proba, loan_amounts) → dict` — Sweep thresholds, find optimal
- `generate_tradeoff_curves(y_true, y_pred_proba, loan_amounts) → Plots` — Approval vs default capture
- `segment_by_risk(y_pred_proba, segment_thresholds) → pd.DataFrame` — Risk segments

**Dependencies:** pandas, numpy, matplotlib, seaborn

**Output Artifacts:** `reports/business_metrics.json`, tradeoff plots

**Config Reference:** `decision_config.yaml` specifies approval/reject thresholds, loan amount range

**Critical Formula:** Expected Loss = sum(loan_amount_i × P(default)_i) for all approved applications

---

## 7. Inference Module (`src/inference/`)

**Purpose:** Provide real-time scoring interface via API or Dashboard.

**Responsibilities:**
- Load all artifacts (model, calibrator, feature transformer, SHAP explainer)
- Accept application data (new loan request)
- Apply feature transformations
- Generate risk score
- Apply decision policy
- Generate SHAP explanations
- Return score, decision, top feature contributions

**Key Functions:**
- `load_production_pipeline() → ProductionPipeline` — Load all artifacts
- `score_application(application_data: dict) → dict` — End-to-end scoring
  - Input: {age, income, credit_score, ...}
  - Output: {risk_score, decision, top_features}
- `generate_explanation(model, shap_explainer, X_sample) → dict` — SHAP explanations

**API Endpoints (if FastAPI):**
- `POST /score` — Single application scoring
- `POST /batch-score` — Batch scoring

**Dashboard (if Streamlit):**
- Input form for applicant details
- Real-time risk score display
- Decision output
- Top feature importance plot

**Dependencies:** pandas, numpy, scikit-learn, shap, fastapi (or streamlit)

**Output Artifacts:** Deployed API or Dashboard service

---

## 8. Utils Module (`src/utils/`)

**Purpose:** Shared utilities, logging, config management.

**Responsibilities:**
- Load and parse YAML configs
- Set up structured logging (console + file)
- Provide common helper functions

**Key Functions:**
- `load_config(path: str) → dict` — Load YAML config
- `setup_logging(log_dir: str, level: str) → Logger` — Configure logging
- `set_random_seed(seed: int)` — Set numpy/sklearn/tf seeds
- `create_directories(paths: list)` — Create required dirs

**Dependencies:** pyyaml, logging

**Output Artifacts:** Logs in `logs/` directory

---

## Module Dependencies

```
Data Ingestion
  ↓
  ├─→ EDA (exploratory)
  ├─→ Feature Engineering
        ↓
        ↓
      Modeling
        ↓
        ↓
      Evaluation
        ↓
        ↓
      Decisioning
        ↓
        ↓
      Inference
```

**Sequential Pipeline:**
1. Ingest data
2. Run EDA (exploratory, no impact on model)
3. Engineer features
4. Train models
5. Evaluate
6. Define decisions
7. Deploy inference

---

## Data Flow Summary

| Module | Input | Output | Key Process |
|--------|-------|--------|-------------|
| Ingestion | Raw CSV | train/val/test splits | Load, validate, split |
| EDA | Splits | HTML report | Analyze patterns |
| Features | train/val/test | Transformers + features | Encode, scale, select |
| Modeling | Features + target | Trained models | Train + tune |
| Evaluation | Predictions + target | Metrics + report | Evaluate, calibrate |
| Decisioning | Predictions | Decisions + business metrics | Map to decisions, simulate |
| Inference | New application | Risk score + decision | Real-time scoring |
