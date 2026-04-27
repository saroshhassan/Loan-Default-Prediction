# Troubleshooting Guide

## Data & Ingestion Issues

### Issue: "FileNotFoundError: data/raw/data.csv not found"

**Cause:** Dataset not in correct location

**Solutions:**
1. Verify Kaggle dataset is downloaded:
   ```bash
   ls -la data/raw/
   ```
2. Manually download from Kaggle website, save to `data/raw/`
3. Check `config.yaml` has correct path:
   ```yaml
   paths:
     data_raw: "data/raw"
   ```

---

### Issue: "Column 'LoanApproved' not found in dataset"

**Cause:** Target column name doesn't match

**Solutions:**
1. Check column names in dataset:
   ```python
   import pandas as pd
   df = pd.read_csv("data/raw/your_data.csv")
   print(df.columns)
   ```
2. Update `feature_config.yaml`:
   ```yaml
   data:
     target_column: "correct_column_name"
   ```

---

### Issue: "Missing values exceed 50% in column X"

**Cause:** Data quality issue

**Solutions:**
1. Remove problematic column (if not critical):
   ```yaml
   # feature_config.yaml
   numerical_features:
     - Age
     # - ProblematicColumn  # Comment out
   ```
2. Use different imputation strategy:
   ```yaml
   missing_values:
     strategy: "drop_row"  # Instead of "impute"
   ```

---

### Issue: "Class imbalance ratio is 99:1 (highly imbalanced)"

**Cause:** Target variable is very skewed

**Solutions:**
1. Use SMOTE instead of class weights:
   ```yaml
   # model_config.yaml
   imbalance:
     strategy: "smote"
   ```
2. Collect more data for minority class
3. Consider threshold-based evaluation instead of accuracy

---

## EDA Issues

### Issue: "HTML report not generated (eda_report.html missing)"

**Cause:** EDA module failed silently

**Solutions:**
1. Check logs:
   ```bash
   tail -f logs/eda.log
   ```
2. Run EDA in verbose mode:
   ```python
   python main.py --phase=eda --verbose
   ```
3. Check if features are correctly identified:
   ```yaml
   # feature_config.yaml should have proper feature lists
   numerical_features: [...]
   categorical_features: [...]
   ```

---

### Issue: "Warning: Feature has near-zero variance"

**Cause:** Feature is constant or near-constant (all values ~same)

**Solutions:**
1. Remove the feature:
   ```yaml
   # feature_config.yaml - remove from lists
   ```
2. Feature is expected, but set importance threshold high:
   ```yaml
   selection:
     importance_threshold: 0.05  # Higher = stricter
   ```

---

## Feature Engineering Issues

### Issue: "Data leakage detected! Transformer fit on train, but applied to train again"

**Cause:** Double-encoding of training data

**Solutions:**
1. Verify feature pipeline fits only once:
   ```python
   # Correct:
   transformer.fit(X_train, y_train)
   X_train_transformed = transformer.transform(X_train)
   
   # Wrong:
   transformer.fit(X_train, y_train)
   transformer.fit(X_train, y_train)  # Don't refit!
   ```

---

### Issue: "ValueError: unknown category in feature 'Education'"

**Cause:** Test set has unseen category value

**Solutions:**
1. Handle unseen categories in encoding:
   ```yaml
   encoding:
     handle_unseen: "unknown"  # or "drop"
   ```
2. Use target encoding instead of one-hot:
   ```yaml
   encoding:
     categorical_strategy: "target"
   ```

---

### Issue: "Feature engineering is very slow"

**Cause:** Too many features being created

**Solutions:**
1. Disable unnecessary binning:
   ```yaml
   binning:
     age:
       enabled: false  # Skip age binning
   ```
2. Reduce number of features before scaling:
   ```yaml
   selection:
     keep_top_k: 15  # Was 25
   ```
3. Use parallel processing:
   ```yaml
   # config.yaml
   execution:
     n_jobs: -1  # Use all CPU cores
   ```

---

## Modeling Issues

### Issue: "ROC-AUC = 0.5 (model is random)"

**Cause:** No predictive signal in features

**Solutions:**
1. Check EDA for signal:
   - Are features different between approved/rejected?
   - Check `eda_report.html` for distributions
2. Verify target variable is correct:
   ```python
   print(y_train.value_counts())  # Should be mostly 0, some 1
   ```
3. Check for target leakage (feature that already encodes target)
4. Verify feature engineering was applied:
   ```python
   print(X_train.columns)  # Should have 20+ features
   ```

---

### Issue: "Training takes > 30 minutes"

**Cause:** Hyperparameter tuning is expensive

**Solutions:**
1. Reduce number of tuning trials:
   ```yaml
   tuning:
     n_trials: 50  # Was 100
   ```
2. Disable tuning:
   ```yaml
   tuning:
     enabled: false
   ```
3. Use faster model (LightGBM vs XGBoost):
   ```yaml
   models_to_train:
     - "lightgbm"  # Skip xgboost
   ```

---

### Issue: "Validation ROC-AUC good (0.85), but test ROC-AUC bad (0.65)"

**Cause:** Overfitting on validation set

**Solutions:**
1. Increase regularization:
   ```yaml
   xgboost:
     params:
       reg_alpha: 5  # Was 0
       reg_lambda: 5  # Was 1
   ```
2. Reduce model complexity:
   ```yaml
   xgboost:
     params:
       max_depth: 4  # Was 6
       n_estimators: 50  # Was 100
   ```
3. Increase training data (if possible)

---

### Issue: "Memory error: Cannot allocate 4GB for SMOTE"

**Cause:** Dataset too large for SMOTE

**Solutions:**
1. Use class weights instead:
   ```yaml
   imbalance:
     strategy: "class_weights"
   ```
2. Reduce dataset size by sampling:
   ```python
   df = df.sample(frac=0.5, random_state=42)
   ```

---

## Evaluation Issues

### Issue: "Calibration curve shows model is uncalibrated"

**Cause:** Predicted probabilities don't match actual frequencies

**Solutions:**
1. Apply Platt scaling:
   ```python
   from sklearn.calibration import CalibratedClassifierCV
   calibrated_model = CalibratedClassifierCV(model, method='sigmoid')
   ```
2. Use probability from Logistic Regression as baseline:
   ```yaml
   models_to_train:
     - "baseline"  # Already calibrated
   ```

---

### Issue: "SHAP explainer is very slow"

**Cause:** SHAP computation is expensive for large datasets

**Solutions:**
1. Sample test set:
   ```python
   X_test_sample = X_test.sample(n=100, random_state=42)
   shap_values = explainer.shap_values(X_test_sample)
   ```
2. Use KernelExplainer instead of TreeExplainer (slower but works for any model)

---

## Decisioning Issues

### Issue: "Approval rate is 95% (too lenient)"

**Cause:** Thresholds are too permissive

**Solutions:**
1. Tighten approval threshold:
   ```yaml
   thresholds:
     approve: 0.25  # Was 0.30
   ```
2. Tighten reject threshold:
   ```yaml
   thresholds:
     reject: 0.60  # Was 0.70
   ```

---

### Issue: "Approval rate is 20% (too strict)"

**Cause:** Thresholds are too strict

**Solutions:**
1. Loosen approval threshold:
   ```yaml
   thresholds:
     approve: 0.40  # Was 0.30
   ```
2. Run threshold optimization to find sweet spot

---

### Issue: "Expected loss exceeds budget"

**Cause:** Too many high-risk applicants being approved

**Solutions:**
1. Tighten approval threshold (see above)
2. Check if risk scores are calibrated
3. Reduce loan amounts for high-risk applicants:
   ```python
   # In decisioning logic
   if risk_score > 0.50:
       loan_amount = min(loan_amount, 100000)  # Cap at $100K
   ```

---

## Inference Issues

### Issue: "API throws 500 error on some requests"

**Cause:** Input validation failing or unexpected feature value

**Solutions:**
1. Check request format matches schema:
   ```bash
   curl http://localhost:8000/docs  # View expected schema
   ```
2. Verify Age is 18-100, Income > 0, etc.
3. Check for NaN or null values in request
4. Enable debug logging:
   ```yaml
   logging:
     level: "DEBUG"
   ```

---

### Issue: "API response time > 2 seconds"

**Cause:** Model loading or feature transformation slow

**Solutions:**
1. Profile the API:
   ```python
   python -m cProfile -s cumtime app/api.py
   ```
2. Cache model in memory (already done)
3. Simplify model (fewer trees, lower max_depth)
4. Use model quantization:
   ```bash
   pip install onnx onnxruntime
   # Convert model to ONNX format
   ```

---

### Issue: "Streamlit dashboard crashes on upload"

**Cause:** File too large or format issue

**Solutions:**
1. Limit file size:
   ```python
   # In streamlit_app.py
   if uploaded_file.size > 10_000_000:  # 10MB
       st.error("File too large")
   ```
2. Ensure CSV format (not Excel)
3. Add error handling:
   ```python
   try:
       df = pd.read_csv(uploaded_file)
   except Exception as e:
       st.error(f"Error reading file: {e}")
   ```

---

## General Issues

### Issue: "Random seed not working (not reproducible)"

**Cause:** Seed set after some random operations

**Solutions:**
1. Set seed early in execution:
   ```python
   import numpy as np
   import random
   from sklearn.utils import check_random_state
   
   seed = 42
   random.seed(seed)
   np.random.seed(seed)
   # Set seed BEFORE imports of sklearn/xgboost/lightgbm
   ```

---

### Issue: "Config file not found"

**Cause:** YAML path incorrect

**Solutions:**
1. Check path relative to where Python is running:
   ```bash
   pwd  # Print working directory
   ls configs/  # Verify files exist
   ```
2. Use absolute path in config:
   ```python
   config_path = os.path.join(os.getcwd(), "configs/config.yaml")
   ```

---

### Issue: "Dependency version conflicts"

**Cause:** Incompatible package versions

**Solutions:**
1. Recreate virtual environment:
   ```bash
   deactivate
   rm -rf venv/
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
2. Pin specific versions in requirements.txt:
   ```
   scikit-learn==1.0.2
   xgboost==1.5.0
   lightgbm==3.3.1
   ```

---

## Getting Help

If issue not in this guide:

1. **Check logs:**
   ```bash
   tail -f logs/*.log
   ```

2. **Run with verbose output:**
   ```bash
   python main.py --verbose
   ```

3. **Check configuration:**
   ```bash
   python -c "from src.utils.config_loader import load_config; print(load_config('configs/config.yaml'))"
   ```

4. **Verify dependencies:**
   ```bash
   pip list
   ```

5. **Create GitHub issue** with:
   - Full error message
   - Python version
   - Command that failed
   - Last 50 lines of relevant log
