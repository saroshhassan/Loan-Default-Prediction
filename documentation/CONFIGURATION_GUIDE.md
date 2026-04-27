# Configuration Guide

## Overview

All system behavior is controlled via YAML configuration files in `configs/`. You can change almost everything without touching code.

---

## Main Configuration File: `config.yaml`

**Location:** `configs/config.yaml`

**Purpose:** Global settings (paths, logging, random seeds, data locations)

```yaml
# System Paths
paths:
  project_root: "."
  data_raw: "data/raw"
  data_processed: "data/processed"
  models: "models"
  reports: "reports"
  logs: "logs"

# Random Seed (for reproducibility)
random_seed: 42

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
  save_to_file: true
  log_dir: "logs"

# Data Loading
data:
  dataset_name: "credit_risk_loan_default"
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
  stratify_by: "LoanApproved"  # Column to stratify on
  random_state: 42

# Kaggle Integration
kaggle:
  enabled: true
  dataset_name: "algozee/credit-risk-and-loan-default-analysis-dataset"
  download_dir: "data/raw"
  force_download: false  # Set true to re-download

# Execution
execution:
  verbose: true
  n_jobs: -1  # -1 = use all CPU cores
```

**When to Modify:**
- Change `random_seed` if reproducibility not required
- Change `logging.level` to DEBUG if troubleshooting
- Change `kaggle.dataset_name` if using different dataset
- Change `data.train_split` if using different proportions (but keep sum = 1.0)

---

## Feature Configuration: `feature_config.yaml`

**Location:** `configs/feature_config.yaml`

**Purpose:** Define how features are engineered, encoded, and selected

```yaml
# Data Schema
data:
  target_column: "LoanApproved"
  numerical_features:
    - Age
    - Income
    - CreditScore
    - LoanAmount
    - YearsExperience
  categorical_features:
    - Gender
    - Education
    - City
    - EmploymentType

# Missing Value Handling
missing_values:
  strategy: "impute"  # impute, drop_row, drop_column
  numerical_method: "median"  # median, mean
  categorical_method: "mode"  # mode, constant
  categorical_constant: "Unknown"

# Outlier Detection
outliers:
  enabled: true
  method: "iqr"  # iqr, zscore
  iqr_multiplier: 1.5
  zscore_threshold: 3.0
  action: "flag"  # flag, cap, drop

# Feature Engineering
binning:
  enabled: true
  age:
    enabled: true
    bins: [0, 25, 35, 50, 100]
    labels: ["18-25", "25-35", "35-50", "50+"]
  income:
    enabled: true
    bins: [0, 35000, 75000, 1000000]
    labels: ["low", "mid", "high"]
  credit_score:
    enabled: true
    bins: [0, 600, 700, 800, 1000]
    labels: ["poor", "fair", "good", "excellent"]
  loan_amount:
    enabled: true
    bins: [0, 100000, 300000, 1000000]
    labels: ["small", "medium", "large"]

# Ratio Features
ratios:
  enabled: true
  debt_to_income:
    numerator: "LoanAmount"
    denominator: "Income"
    enabled: true
  experience_to_age:
    numerator: "YearsExperience"
    denominator: "Age"
    enabled: true

# Encoding Strategy
encoding:
  categorical_strategy: "target"  # target, frequency, one_hot
  ordinal_encoding:
    enabled: true
    Education:
      HighSchool: 1
      Bachelor: 2
      Master: 3
      PhD: 4
  one_hot_encoding:
    enabled: true
    max_categories: 10  # Combine rare categories if > 10
  target_encoding:
    enabled: true
    smoothing: 1.0  # Regularization to prevent overfitting

# Scaling
scaling:
  enabled: true
  method: "standard"  # standard, robust, minmax
  features_to_scale: "numerical"  # numerical, all, custom

# Feature Selection
selection:
  enabled: true
  method: "importance"  # importance, correlation, manual
  importance_threshold: 0.01
  keep_top_k: 25  # Keep top N features by importance
  correlation_threshold: 0.95  # Remove if > 95% correlated
```

**When to Modify:**
- Change `binning` thresholds if age/income brackets don't fit your domain
- Change `encoding.categorical_strategy` from `target` to `one_hot` if encoding instability
- Disable features by setting `enabled: false` (e.g., `credit_score.enabled: false`)
- Change `selection.keep_top_k` to reduce model complexity (faster training)
- Change `outliers.action` from `flag` to `drop` if outliers are data errors

---

## Model Configuration: `model_config.yaml`

**Location:** `configs/model_config.yaml`

**Purpose:** Define models to train, hyperparameters, imbalance handling

```yaml
# Models to Train
models_to_train:
  - "baseline"      # Logistic Regression
  - "xgboost"       # XGBoost
  - "lightgbm"      # LightGBM

# Model Hyperparameters
baseline:
  type: "logistic_regression"
  params:
    C: 1.0
    penalty: "l2"
    solver: "lbfgs"
    max_iter: 1000

xgboost:
  type: "xgboost"
  params:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100
    subsample: 0.8
    colsample_bytree: 0.8
    reg_alpha: 0
    reg_lambda: 1
    scale_pos_weight: null  # Auto-compute from class imbalance

lightgbm:
  type: "lightgbm"
  params:
    max_depth: 7
    learning_rate: 0.1
    n_estimators: 100
    num_leaves: 31
    subsample: 0.8
    colsample_bytree: 0.8
    lambda_l1: 0
    lambda_l2: 1

# Class Imbalance Handling
imbalance:
  strategy: "class_weights"  # smote, class_weights, focal_loss
  smote:
    k_neighbors: 5
    random_state: 42
  class_weights:
    method: "balanced"  # balanced, custom
    custom_weights: null  # {0: 1.0, 1: 3.0} or null for auto

# Hyperparameter Tuning
tuning:
  enabled: true
  algorithm: "optuna"  # optuna, grid_search, random_search
  n_trials: 100  # Number of tuning iterations
  timeout: 3600  # Max time in seconds
  cv_folds: 5
  pruner: "median"  # early stopping strategy
  sampler: "tpe"  # Tree-structured Parzen Estimator
  objectives:
    primary: "roc_auc"  # roc_auc, pr_auc, f1
    constraints: []  # e.g., ["precision > 0.80"]

# Training Settings
training:
  random_seed: 42
  batch_size: 32  # For SGD-based models
  early_stopping_rounds: 10  # For boosting models
  early_stopping_metric: "roc_auc"
  verbose: true

# Cross-Validation
cross_validation:
  enabled: true
  n_splits: 5
  stratified: true

# Model Persistence
persistence:
  save_trained_models: true
  save_best_model_only: false
  model_dir: "models"
  save_hyperparameters: true
```

**When to Modify:**
- Change `models_to_train` to skip models (e.g., remove `"xgboost"` to save time)
- Adjust hyperparameters manually if you don't want automatic tuning:
  ```yaml
  tuning:
    enabled: false
  ```
- Increase `n_trials` from 100 to 200 if tuning takes too long
- Change `imbalance.strategy` to "smote" if you have < 5K samples
- Adjust `class_weights.custom_weights` manually if known:
  ```yaml
  class_weights:
    custom_weights: {0: 1.0, 1: 2.5}  # Reject class is 2.5x more important
  ```

---

## Decision Configuration: `decision_config.yaml`

**Location:** `configs/decision_config.yaml`

**Purpose:** Define decision thresholds and business rules

```yaml
# Decision Thresholds
thresholds:
  approve: 0.30  # Risk score < 0.30 → APPROVE
  reject: 0.70   # Risk score > 0.70 → REJECT
  # Risk score 0.30-0.70 → REVIEW (manual)

# Risk-Based Pricing (Optional)
risk_pricing:
  enabled: true
  tiers:
    low:
      min_score: 0.00
      max_score: 0.25
      interest_rate: 4.5
      description: "Excellent credit, stable employment"
    moderate:
      min_score: 0.25
      max_score: 0.50
      interest_rate: 5.5
      description: "Good credit, some risk factors"
    high:
      min_score: 0.50
      max_score: 0.75
      interest_rate: 7.5
      description: "High risk, conditional approval"
    very_high:
      min_score: 0.75
      max_score: 1.00
      interest_rate: null
      description: "Very high risk, declined"

# Business Constraints
constraints:
  min_approval_rate: 0.50  # At least 50% must be approvable
  max_default_rate: 0.20   # Default rate in approved can't exceed 20%
  max_expected_loss: 500000  # Total expected loss cap

# Risk Segmentation
segments:
  low_risk:
    min_score: 0.00
    max_score: 0.25
    action: "auto_approve"  # auto_approve, manual_review, auto_reject
    requires_documentation: false
  moderate_risk:
    min_score: 0.25
    max_score: 0.50
    action: "manual_review"
    requires_documentation: true
  high_risk:
    min_score: 0.50
    max_score: 0.75
    action: "conditional"  # May approve with conditions (co-signer, deposit, etc.)
    requires_documentation: true
  very_high_risk:
    min_score: 0.75
    max_score: 1.00
    action: "auto_reject"
    requires_documentation: false

# Audit & Logging
audit:
  log_all_decisions: true
  log_file: "logs/decisions.log"
  track_fairness: true
  fairness_by_demographic: ["Gender", "City"]

# Threshold Optimization
optimization:
  auto_tune: false  # Set true to find optimal thresholds
  objective: "balance"  # balance, maximize_approval, minimize_loss
```

**When to Modify:**
- Change `approve` threshold from 0.30 to 0.35 to be more conservative (fewer approvals)
- Change `reject` threshold from 0.70 to 0.60 to auto-reject more marginal cases
- Add/remove risk tiers in `risk_pricing.tiers`
- Change `action` for segments (e.g., `"conditional"` for high-risk, allowing special handling)
- Enable `auto_tune: true` to let system find optimal thresholds automatically
- Adjust `constraints` to reflect business requirements

---

## Example: Adjusting for Conservative Lending

**Scenario:** Want to reduce default risk even if it means fewer approvals

**Changes:**

```yaml
# decision_config.yaml
thresholds:
  approve: 0.25  # ↓ More conservative (was 0.30)
  reject: 0.65   # ↓ More aggressive rejection (was 0.70)

constraints:
  max_default_rate: 0.10  # ↓ Stricter limit (was 0.20)

# model_config.yaml
imbalance:
  class_weights:
    custom_weights: {0: 1.0, 1: 3.0}  # ↑ Penalize rejections more
```

**Expected Impact:**
- Approval rate: 78% → 55% (fewer approvals)
- Default rate: 12% → 8% (safer portfolio)
- Expected loss: $180K → $100K (less risk)

---

## Example: Adjusting for Growth

**Scenario:** Want to grow loan volume, accept higher risk

**Changes:**

```yaml
# decision_config.yaml
thresholds:
  approve: 0.40  # ↑ More lenient (was 0.30)
  reject: 0.80   # ↑ More lenient rejection (was 0.70)

constraints:
  min_approval_rate: 0.75  # ↑ Target high approval (was 0.50)
  max_default_rate: 0.25   # ↑ Accept higher defaults (was 0.20)
```

**Expected Impact:**
- Approval rate: 78% → 88% (more approvals, more volume)
- Default rate: 12% → 18% (higher risk)
- Revenue: Increases, but so does loss

---

## Validation Checklist

Before running pipeline with new configs:

- [ ] `random_seed` set consistently (if reproducibility needed)
- [ ] `target_column` matches your dataset's target
- [ ] `numerical_features` and `categorical_features` match your data
- [ ] Binning `bins` are in ascending order
- [ ] `train_split + val_split + test_split = 1.0`
- [ ] `approve_threshold < reject_threshold`
- [ ] Risk pricing `interest_rates` are realistic for your domain
- [ ] `min_approval_rate + max_default_rate` are feasible together

---

## Loading Configs Programmatically

In Python code:

```python
from src.utils.config_loader import load_config

# Load all configs
config = load_config("configs/config.yaml")
feature_config = load_config("configs/feature_config.yaml")
model_config = load_config("configs/model_config.yaml")
decision_config = load_config("configs/decision_config.yaml")

# Access values
target_column = feature_config['data']['target_column']
approve_threshold = decision_config['thresholds']['approve']
```

---

## Default Values Reference

| Config | Parameter | Default | Type |
|--------|-----------|---------|------|
| config.yaml | random_seed | 42 | int |
| feature_config.yaml | outlier method | "iqr" | str |
| feature_config.yaml | encoding strategy | "target" | str |
| model_config.yaml | imbalance strategy | "class_weights" | str |
| model_config.yaml | tuning n_trials | 100 | int |
| decision_config.yaml | approve_threshold | 0.30 | float |
| decision_config.yaml | reject_threshold | 0.70 | float |
