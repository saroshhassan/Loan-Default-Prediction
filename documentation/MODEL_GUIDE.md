# Model Guide: Training, Tuning & Selection

## Model Selection Strategy

This system trains **two types of models**:

1. **Baseline Model:** Logistic Regression (interpretable, fast)
2. **Advanced Models:** XGBoost + LightGBM (high accuracy, captures non-linearity)

---

## 1. Baseline Model: Logistic Regression

### Why Baseline?
- **Interpretable:** Model output = log-odds, coefficients are directly interpretable
- **Fast:** Trains in milliseconds
- **Calibrated:** Outputs are naturally calibrated probabilities
- **Regulatory:** Easier to explain to non-technical stakeholders

### Usage
Use baseline for:
- Initial risk scoring (quick decisions)
- Understanding feature impact (high coefficient = strong signal)
- Regulatory/compliance scenarios

### Performance Expectation
- Typical ROC-AUC: **0.65-0.75**
- Calibration: **Excellent** (built-in)

### Configuration

```yaml
model_config.yaml:
  baseline:
    type: "logistic_regression"
    params:
      C: 1.0  # Inverse regularization strength
      penalty: "l2"  # L1 or L2 regularization
      max_iter: 1000
      solver: "lbfgs"
```

---

## 2. Advanced Models: XGBoost & LightGBM

### Why Advanced?
- **Non-linear relationships:** Captures complex feature interactions
- **High accuracy:** ROC-AUC typically 0.80-0.90
- **Feature importance:** Built-in importance scores
- **Robustness:** Handles outliers and categorical features well

### XGBoost

**Strengths:**
- Most accurate (usually best ROC-AUC)
- Stable, well-tested in production
- Excellent SHAP explanations

**Weaknesses:**
- Slower training (especially with large datasets)
- More hyperparameters to tune

**Typical Performance:**
- ROC-AUC: **0.80-0.88**
- Training time: 1-5 minutes (depending on dataset size)

**Configuration:**

```yaml
model_config.yaml:
  xgboost:
    type: "xgboost"
    params:
      max_depth: 6
      learning_rate: 0.1
      n_estimators: 100
      subsample: 0.8
      colsample_bytree: 0.8
      reg_alpha: 0  # L1 regularization
      reg_lambda: 1  # L2 regularization
```

### LightGBM

**Strengths:**
- Fastest training (good for large datasets)
- Memory-efficient
- Handles categorical features natively

**Weaknesses:**
- Can overfit on small datasets
- Less mature than XGBoost in some production environments

**Typical Performance:**
- ROC-AUC: **0.78-0.86**
- Training time: <1 minute (even on large datasets)

**Configuration:**

```yaml
model_config.yaml:
  lightgbm:
    type: "lightgbm"
    params:
      max_depth: 7
      learning_rate: 0.1
      n_estimators: 100
      num_leaves: 31
      subsample: 0.8
      colsample_bytree: 0.8
      lambda_l1: 0  # L1 regularization
      lambda_l2: 1  # L2 regularization
```

---

## 3. Class Imbalance Handling

Loan default datasets are typically **imbalanced** (e.g., 80% approved, 20% rejected).

### Why Handle Imbalance?

Without handling, models learn:
- High accuracy by predicting "approved" for everything
- Poor sensitivity (miss actual defaults)
- Uncalibrated probabilities

### Strategies

#### Strategy 1: SMOTE (Synthetic Minority Oversampling)

**How:** Create synthetic examples of minority class (rejected applicants) via interpolation.

**When:** Use if dataset is small (<50K rows)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**Pros:**
- Improves recall (catches more defaults)
- Works well for highly imbalanced data

**Cons:**
- Can cause overfitting
- Creates synthetic data (may not reflect reality)

#### Strategy 2: Class Weights

**How:** Penalize misclassifying the minority class more heavily.

**When:** Use if dataset is large (>50K rows)

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
# If 20% rejected, rejected_weight ≈ 2.5x, approved_weight ≈ 0.5x
```

**Pros:**
- No synthetic data needed
- Works well with gradient boosting
- Prevents overfitting

**Cons:**
- Requires tuning weight ratios

#### Strategy 3: Focal Loss (Advanced)

**How:** Reduce weight of easy examples, focus on hard examples.

**When:** Use for highly imbalanced datasets (< 5% minority)

**Implementation:** Not natively supported; requires custom loss function.

### Recommendation

**For this system:**
- **If dataset < 10K rows:** Use SMOTE
- **If dataset 10K-100K rows:** Use class weights
- **If dataset > 100K rows:** Use class weights or focal loss

**Default Configuration:**

```yaml
model_config.yaml:
  imbalance_strategy: "class_weights"  # or smote, focal_loss
  smote_k_neighbors: 5
  class_weight_ratio: "balanced"  # Auto-compute or specify manually
```

---

## 4. Hyperparameter Tuning

The system uses **Optuna** to find optimal hyperparameters.

### Tuning Strategy

**Objective:** Maximize ROC-AUC on validation set

**Search Space:**

| Parameter | Range | Type |
|-----------|-------|------|
| max_depth | 3-15 | Integer |
| learning_rate | 0.001-0.5 | Float (log) |
| n_estimators | 50-500 | Integer |
| subsample | 0.5-1.0 | Float |
| colsample_bytree | 0.5-1.0 | Float |
| reg_alpha (L1) | 0-10 | Float |
| reg_lambda (L2) | 0-10 | Float |

### Tuning Configuration

```yaml
model_config.yaml:
  tuning:
    enabled: true
    study_name: "loan_default_tuning"
    n_trials: 100  # Number of iterations
    timeout: 3600  # Max time (seconds)
    cv_folds: 5  # Cross-validation folds
    pruner: "median"  # Early stopping strategy
    sampler: "tpe"  # Tree-structured Parzen Estimator
```

### Tuning Workflow

```
1. Initialize Optuna study
2. For each trial:
   a) Sample hyperparameters from search space
   b) Train model with those hyperparameters
   c) Evaluate on validation set (ROC-AUC)
   d) Report score back to Optuna
   e) Prune trial if score is unlikely to improve
3. Return best hyperparameters found
```

### Output

Best hyperparameters saved to `models/best_params.json`:

```json
{
  "max_depth": 7,
  "learning_rate": 0.08,
  "n_estimators": 200,
  "subsample": 0.85,
  "colsample_bytree": 0.9,
  "reg_alpha": 1.2,
  "reg_lambda": 5.0,
  "roi_auc_validation": 0.854
}
```

---

## 5. Model Evaluation Metrics

### Primary Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **ROC-AUC** | Area under receiver operating characteristic curve | Probability that model ranks random default higher than random approved; [0, 1], 0.5 = random, 1.0 = perfect |
| **PR-AUC** | Area under precision-recall curve | Summary of precision-recall tradeoff; better for imbalanced; [0, 1] |
| **Brier Score** | Mean squared error between predicted & actual | Calibration metric; [0, 1], 0 = perfect |

### Secondary Metrics (at Different Thresholds)

For threshold t in {0.3, 0.5, 0.7}:

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Precision** | TP / (TP + FP) | Of predicted defaults, % actually default |
| **Recall** | TP / (TP + FN) | Of actual defaults, % predicted correctly |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean; balances precision & recall |

### Calibration Metrics

| Metric | Target | Interpretation |
|--------|--------|-----------------|
| **Expected Calibration Error (ECE)** | < 0.05 | Average difference between predicted & observed probability |
| **Maximum Calibration Error** | < 0.15 | Maximum difference across probability bins |

---

## 6. Training Workflow

### Step 1: Prepare Data

```python
# Load data
X_train, y_train = load_data("data/processed/train.csv")
X_val, y_val = load_data("data/processed/val.csv")

# Apply feature engineering (fit on train only)
transformer = FeatureEngineeringPipeline()
transformer.fit(X_train, y_train)
X_train_transformed = transformer.transform(X_train)
X_val_transformed = transformer.transform(X_val)
```

### Step 2: Train Baseline

```python
baseline_model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
baseline_model.fit(X_train_transformed, y_train)

# Evaluate
y_pred_proba = baseline_model.predict_proba(X_val_transformed)[:, 1]
roc_auc = roc_auc_score(y_val, y_pred_proba)
print(f"Baseline ROC-AUC: {roc_auc:.4f}")
```

### Step 3: Tune Advanced Models

```python
# Tune XGBoost
best_params_xgb = tune_xgboost(X_train_transformed, y_train, X_val_transformed, y_val)

# Tune LightGBM
best_params_lgb = tune_lightgbm(X_train_transformed, y_train, X_val_transformed, y_val)
```

### Step 4: Train Final Models

```python
xgb_model = XGBClassifier(**best_params_xgb)
xgb_model.fit(X_train_transformed, y_train)

lgb_model = LGBMClassifier(**best_params_lgb)
lgb_model.fit(X_train_transformed, y_train)
```

### Step 5: Select Best Model

Compare on **test set** (not used during tuning):

```python
X_test, y_test = load_data("data/processed/test.csv")
X_test_transformed = transformer.transform(X_test)

models = {
    'baseline': baseline_model,
    'xgboost': xgb_model,
    'lightgbm': lgb_model
}

for name, model in models.items():
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"{name} ROC-AUC: {roc_auc:.4f}")
```

---

## 7. Model Persistence

### Saving Models

```python
import pickle

# Save trained model
with open("models/xgboost_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

# Save feature transformer
with open("models/feature_transformer.pkl", "wb") as f:
    pickle.dump(transformer, f)

# Save best hyperparameters
import json
with open("models/best_params_xgb.json", "w") as f:
    json.dump(best_params_xgb, f)
```

### Loading Models

```python
# Load model
with open("models/xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# Use for inference
y_pred_proba = xgb_model.predict_proba(X_new)[:, 1]
```

---

## 8. Model Interpretability

### Feature Importance

```python
# Tree-based importance
importances = xgb_model.feature_importances_
feature_names = transformer.get_feature_names_out()

# Plot
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('reports/feature_importance.png')
```

### SHAP Explanations

```python
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_transformed)

# Plot summary
shap.summary_plot(shap_values, X_test_transformed, plot_type='bar')
plt.savefig('reports/shap_summary.png')
```

---

## 9. Common Issues & Solutions

### Issue: Model Always Predicts One Class

**Cause:** Severe class imbalance not handled; model takes easy path

**Solution:** Enable SMOTE or class weights in config

### Issue: ROC-AUC is 0.5 (Random)

**Cause:** Features have no signal; data quality issue

**Solution:** Review EDA report, check for missing/constant features

### Issue: Validation ROC-AUC High, Test ROC-AUC Low

**Cause:** Overfitting; model memorized validation set

**Solution:** Increase regularization (reg_alpha, reg_lambda) in config

### Issue: Training is Very Slow

**Cause:** Dataset too large or hyperparameter space too big

**Solution:** Reduce n_trials in tuning config; use LightGBM instead of XGBoost
