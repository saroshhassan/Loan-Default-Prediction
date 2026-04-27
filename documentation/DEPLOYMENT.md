# Deployment Guide

## Overview

This guide walks through deploying the system end-to-end, from environment setup to serving predictions.

---

## Prerequisites

- Python 3.8+
- pip or conda
- Kaggle API credentials (for data download)
- ~2GB disk space (for model artifacts and data)

---

## Step 1: Clone/Setup Project

```bash
# Navigate to project
cd d:/My\ Projects/RiskAssessmentManager

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

---

## Step 2: Install Dependencies

```bash
# Install all packages
pip install -r requirements.txt

# Verify installations
python -c "import pandas; import xgboost; import lightgbm; print('All packages installed!')"
```

---

## Step 3: Configure Kaggle API

```bash
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/
# 4. Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

**Verify:**
```bash
kaggle datasets list
```

---

## Step 4: Download Dataset

```bash
# Download from Kaggle
python -m src.data_ingestion.loader --download

# Or manually:
kaggle datasets download -d algozee/credit-risk-and-loan-default-analysis-dataset -p data/raw/
```

---

## Step 5: Run Full Pipeline

```bash
# Option A: Run all phases sequentially
python main.py --pipeline

# Option B: Run specific phase
python main.py --phase=data_ingestion
python main.py --phase=eda
python main.py --phase=feature_engineering
python main.py --phase=modeling
python main.py --phase=evaluation
python main.py --phase=decisioning
python main.py --phase=inference
```

---

## Step 6: Start Inference Service

### Option A: FastAPI

```bash
# Terminal 1: Start API
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Test
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "Income": 65000,
    "CreditScore": 750,
    "LoanAmount": 200000,
    "YearsExperience": 10,
    "Gender": "M",
    "Education": "Bachelor",
    "City": "NewYork",
    "EmploymentType": "Employed"
  }'

# View API docs
open http://localhost:8000/docs
```

### Option B: Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py

# Automatically opens http://localhost:8501
```

---

## Step 7: Monitor & Maintain

### View Logs

```bash
# API logs
tail -f logs/api_requests.log

# Inference logs
tail -f logs/inference.log

# Model training logs
tail -f logs/training.log
```

### Model Retraining (Weekly/Monthly)

```bash
# Retrain models with new data
python main.py --phase=modeling

# This:
# 1. Reloads training data
# 2. Retrains baseline + advanced models
# 3. Saves new model artifacts to models/

# Don't stop API; it uses cached model until manually reloaded
```

### Updating Thresholds (No Retraining)

```bash
# Edit decision_config.yaml
# Change approve_threshold, reject_threshold

# Restart API to pick up new thresholds
# (FastAPI auto-reloads if running with --reload)
```

---

## Production Deployment

### Using Gunicorn + Nginx

```bash
# Install gunicorn
pip install gunicorn

# Start with gunicorn (multiple workers for concurrency)
gunicorn -w 4 -b 0.0.0.0:8000 app.api:app

# Nginx reverse proxy (nginx.conf):
upstream loan_api {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://loan_api;
        proxy_set_header Host $host;
    }
}
```

### Using Docker

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Download model & data
RUN python -m src.data_ingestion.loader --download
RUN python main.py --phase=modeling

# Expose API
EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build image
docker build -t loan-risk-api:latest .

# Run container
docker run -p 8000:8000 loan-risk-api:latest
```

---

## Scaling Considerations

### For High Throughput (1000+ requests/day)

1. **Load Balancer:** Use nginx or AWS ELB
2. **Multiple Workers:** `gunicorn -w 8 -b 0.0.0.0:8000 app.api:app`
3. **Model Caching:** Keep model in memory (already done)
4. **Database:** Add PostgreSQL to store decisions for audit trail

### For Real-Time Requirements (<100ms per request)

1. **Model Optimization:** Use ONNX or TensorRT for faster inference
2. **Feature Caching:** Pre-compute aggregation features
3. **CPU Affinity:** Pin API processes to specific cores

### For Regulatory Compliance

1. **Audit Logging:** Log all decisions with timestamps
2. **Model Versioning:** Track which model version made each decision
3. **Explainability:** Store SHAP values for every prediction
4. **Access Control:** Use OAuth 2.0 or similar for API authentication

---

## Continuous Integration/Deployment (CI/CD)

### GitHub Actions Example

```yaml
# .github/workflows/deploy.yml
name: Test & Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest tests/
      - run: python main.py --phase=modeling
      
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: success()
    steps:
      - uses: actions/checkout@v2
      - run: docker build -t loan-risk-api:latest .
      - run: docker push your_registry/loan-risk-api:latest
      - run: kubectl set image deployment/loan-api api=your_registry/loan-risk-api:latest
```

---

## Monitoring in Production

### Key Metrics to Track

```python
# In app/monitoring.py
metrics = {
    'requests_per_minute': 0,
    'avg_response_time_ms': 0,
    'approval_rate': 0,
    'rejection_rate': 0,
    'review_rate': 0,
    'model_drift': 0,  # Actual default rate vs predicted
    'api_errors': 0,
    'cache_hit_rate': 0,
}
```

### Alerting Rules

```yaml
# alerts.yaml
rules:
  - name: high_api_latency
    condition: "avg_response_time_ms > 500"
    action: "page_oncall"
  
  - name: model_drift_detected
    condition: "model_drift > 0.10"
    action: "notify_data_science_team"
  
  - name: approval_rate_anomaly
    condition: "approval_rate < 0.50 OR approval_rate > 0.95"
    action: "notify_business_team"
```

---

## Troubleshooting Deployment

### API Won't Start

```bash
# 1. Check Python version
python --version  # Should be 3.8+

# 2. Check dependencies
pip list | grep -E "fastapi|uvicorn"

# 3. Check port is free
netstat -an | grep 8000

# 4. Start with verbose logging
uvicorn app.api:app --reload --log-level debug
```

### Model Loading Error

```bash
# 1. Verify model file exists
ls -la models/

# 2. Check model format
python -c "import pickle; pickle.load(open('models/xgboost_model.pkl', 'rb'))"

# 3. Retrain if corrupted
python main.py --phase=modeling
```

### Slow Inference (> 500ms)

```bash
# 1. Profile bottleneck
python -m cProfile -s cumtime app/api.py

# 2. Check model size
du -h models/

# 3. Consider model compression (ONNX, TensorRT)
```

---

## Rollback Procedure

If new model performs worse:

```bash
# 1. Check previous model version
ls -lart models/  # See timestamps

# 2. Restore from backup
cp models/xgboost_model.pkl.backup models/xgboost_model.pkl

# 3. Restart API
kill -HUP $(pgrep -f "uvicorn")
```

---

## Performance Tuning Checklist

- [ ] API responds in < 300ms for single request
- [ ] Batch endpoint handles 100+ applications in < 10s
- [ ] Model loading takes < 2 seconds
- [ ] CPU utilization < 80%
- [ ] Memory usage < 2GB
- [ ] No SQL injection vulnerabilities (if using database)
- [ ] Rate limiting enabled (100 req/min per IP)
- [ ] Request logging enabled (audit trail)
- [ ] CORS configured if serving web frontend
- [ ] HTTPS enabled in production (not in dev)

---

## Decommissioning (If Needed)

```bash
# 1. Stop API
pkill -f "uvicorn"

# 2. Archive final decisions log
tar -czf decisions_archive_$(date +%Y%m%d).tar.gz logs/decisions.log

# 3. Backup models for compliance
tar -czf models_archive_$(date +%Y%m%d).tar.gz models/

# 4. Clean up
rm -rf venv/  # But keep data/ and reports/ for audit
```
