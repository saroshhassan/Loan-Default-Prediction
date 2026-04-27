# API Reference & Inference Interface

## Overview

The inference system exposes two interfaces for real-time risk scoring:

1. **FastAPI REST Service** (recommended for production)
2. **Streamlit Dashboard** (recommended for prototyping/visualization)

---

## FastAPI REST Service

### Running the API

```bash
# Terminal 1: Start API server
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Test API
curl http://localhost:8000/docs  # Interactive Swagger docs
```

### Endpoints

#### 1. Health Check

**Endpoint:** `GET /health`

**Purpose:** Verify API is running

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-04-26T10:30:00",
  "version": "1.0.0"
}
```

---

#### 2. Single Application Scoring

**Endpoint:** `POST /score`

**Purpose:** Score a single loan application and return risk score + decision

**Request:**
```bash
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
```

**Response:**
```json
{
  "application_id": "app_20260426_001",
  "risk_score": 0.28,
  "decision": "APPROVE",
  "approval_probability": 0.72,
  "confidence": 0.92,
  "top_features": [
    {
      "feature": "credit_good",
      "contribution": 0.15,
      "direction": "positive"
    },
    {
      "feature": "debt_to_income",
      "contribution": -0.08,
      "direction": "negative"
    },
    {
      "feature": "emp_employed",
      "contribution": 0.12,
      "direction": "positive"
    }
  ],
  "explanations": [
    "Good credit score (750) reduces default risk by 15%",
    "Debt-to-income ratio (3.08x) increases risk by 8%",
    "Stable employment reduces risk by 12%"
  ],
  "suggested_interest_rate": 4.5,
  "loan_conditions": "Standard terms"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `application_id` | String | Unique identifier for this scoring request |
| `risk_score` | Float [0,1] | Probability of default |
| `decision` | Enum | APPROVE, REVIEW, or REJECT |
| `approval_probability` | Float [0,1] | 1 - risk_score |
| `confidence` | Float [0,1] | Model's confidence in this prediction |
| `top_features` | List | Top 3-5 feature contributions via SHAP |
| `explanations` | List | Human-readable explanations |
| `suggested_interest_rate` | Float | Interest rate based on risk tier |
| `loan_conditions` | String | Any special conditions |

**Error Responses:**

```json
{
  "error": "Invalid input",
  "details": "Age must be between 18 and 100"
}
```

---

#### 3. Batch Scoring

**Endpoint:** `POST /batch-score`

**Purpose:** Score multiple applications at once (CSV upload)

**Request:**
```bash
curl -X POST http://localhost:8000/batch-score \
  -F "file=@applications.csv"
```

**Input File (`applications.csv`):**
```csv
Age,Income,CreditScore,LoanAmount,YearsExperience,Gender,Education,City,EmploymentType
35,65000,750,200000,10,M,Bachelor,NewYork,Employed
28,45000,620,100000,5,F,HighSchool,LosAngeles,Employed
45,120000,800,350000,20,M,Master,Chicago,Self-Employed
```

**Response:**
```json
{
  "batch_id": "batch_20260426_0001",
  "total_applications": 3,
  "processed": 3,
  "failed": 0,
  "results": [
    {
      "row_index": 0,
      "risk_score": 0.28,
      "decision": "APPROVE",
      "status": "success"
    },
    {
      "row_index": 1,
      "risk_score": 0.62,
      "decision": "REVIEW",
      "status": "success"
    },
    {
      "row_index": 2,
      "risk_score": 0.15,
      "decision": "APPROVE",
      "status": "success"
    }
  ],
  "download_url": "/downloads/batch_20260426_0001.csv"
}
```

**Output CSV (`batch_20260426_0001.csv`):**
```csv
row_index,risk_score,decision,approval_probability,suggested_interest_rate
0,0.28,APPROVE,0.72,4.5
1,0.62,REVIEW,0.38,6.5
2,0.15,APPROVE,0.85,4.0
```

---

#### 4. Model Explanations (SHAP)

**Endpoint:** `POST /explain`

**Purpose:** Get detailed SHAP explanations for a single application

**Request:**
```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "Income": 65000,
    ...
  }'
```

**Response:**
```json
{
  "application_id": "app_20260426_001",
  "risk_score": 0.28,
  "base_value": 0.35,
  "feature_contributions": [
    {
      "feature": "credit_good",
      "shap_value": 0.15,
      "base_value": 0.35,
      "predicted_value": 0.50,
      "interpretation": "Reduces risk by 0.15 (from baseline 0.35)"
    },
    {
      "feature": "debt_to_income",
      "shap_value": -0.08,
      "base_value": 0.35,
      "predicted_value": 0.42,
      "interpretation": "Increases risk by 0.08"
    }
  ],
  "shap_plot_url": "/plots/shap_20260426_001.png"
}
```

---

#### 5. Model Information

**Endpoint:** `GET /model-info`

**Purpose:** Get metadata about deployed model

**Request:**
```bash
curl http://localhost:8000/model-info
```

**Response:**
```json
{
  "model_type": "XGBoost",
  "model_version": "1.2.0",
  "training_date": "2026-04-20",
  "training_dataset_size": 15000,
  "validation_roc_auc": 0.854,
  "test_roc_auc": 0.849,
  "calibration_ece": 0.032,
  "feature_count": 25,
  "decision_policy": {
    "approve_threshold": 0.30,
    "reject_threshold": 0.70
  },
  "last_update": "2026-04-26T08:00:00"
}
```

---

## Streamlit Dashboard

### Running the Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard opens at `http://localhost:8501`

### Features

#### 1. Single Application Scoring

**Interface:**
- Input form with 9 fields (Age, Income, Credit Score, etc.)
- Real-time scoring on submit
- Risk score gauge (red/yellow/green)
- Decision badge (APPROVE/REVIEW/REJECT)
- Top 3 feature explanations
- Suggested interest rate

**Screenshot (ASCII):**
```
┌─────────────────────────────────────┐
│  LOAN DEFAULT RISK ASSESSMENT       │
├─────────────────────────────────────┤
│                                     │
│  Age: [35_____________]             │
│  Income: [65000_____________]       │
│  Credit Score: [750_____________]   │
│  Loan Amount: [200000_________]     │
│  Years Experience: [10________]     │
│  Gender: [Dropdown: M/F/Other]      │
│  Education: [Dropdown]              │
│  City: [Text Input]                 │
│  Employment: [Dropdown]             │
│                                     │
│  [SCORE APPLICATION] [CLEAR FORM]   │
│                                     │
│  ┌─ RISK SCORE ─┐                  │
│  │    28%       │                  │
│  │ [████░░░░]   │  ← Low Risk      │
│  └──────────────┘                  │
│                                     │
│  Decision: ✅ APPROVE               │
│  Interest Rate: 4.5%                │
│                                     │
│  📊 TOP FEATURES:                   │
│  ├─ Good credit → -15% risk        │
│  ├─ High DTI → +8% risk            │
│  └─ Stable job → -12% risk         │
│                                     │
└─────────────────────────────────────┘
```

#### 2. Batch Upload & Scoring

**Interface:**
- CSV file uploader
- Progress bar during processing
- Results table (rows, scores, decisions)
- Download button for results CSV

#### 3. Threshold Tuner

**Interface:**
- Slider for approve_threshold (0.1 - 0.9)
- Slider for reject_threshold (0.1 - 0.9)
- Real-time update of metrics:
  - Approval rate
  - Default rate (in approved)
  - False positive rate
  - Expected loss
- Tradeoff curve visualization

**Interactive Updates:**
```
Move "Approve Threshold" slider from 0.30 → 0.40:
├─ Approval rate: 78% → 88% (more lenient)
├─ Default rate: 12% → 18% (higher risk)
├─ False positive: 5% → 2%
└─ Expected loss: $180K → $320K (more risk)
```

#### 4. Portfolio Overview

**Widgets:**
- Total applications scored (today, this week, all-time)
- Approval rate gauge
- Default rate in approved portfolio
- Portfolio quality chart (risk distribution)
- Recent decisions (last 10 scored)

#### 5. Model Performance Dashboard

**Widgets:**
- ROC curve plot
- Calibration curve
- Feature importance bar chart
- Decision distribution pie chart
- Key metrics cards (ROC-AUC, Precision, Recall)

---

## Request/Response Schema

### Input Schema (Pydantic)

```python
from pydantic import BaseModel, Field

class ApplicationRequest(BaseModel):
    Age: int = Field(..., ge=18, le=100)
    Income: float = Field(..., gt=0)
    CreditScore: int = Field(..., ge=0, le=1000)
    LoanAmount: float = Field(..., gt=0)
    YearsExperience: int = Field(..., ge=0)
    Gender: str = Field(..., pattern="^(M|F|Other)$")
    Education: str = Field(...)
    City: str = Field(...)
    EmploymentType: str = Field(...)
```

### Output Schema (Pydantic)

```python
class ScoringResponse(BaseModel):
    application_id: str
    risk_score: float
    decision: str  # "APPROVE", "REVIEW", "REJECT"
    approval_probability: float
    confidence: float
    top_features: List[FeatureContribution]
    explanations: List[str]
    suggested_interest_rate: float
    loan_conditions: str
```

---

## Error Handling

### Common Errors

| Status | Error | Cause | Solution |
|--------|-------|-------|----------|
| 400 | Invalid input | Missing/invalid field | Check field types & constraints |
| 422 | Validation error | Value outside allowed range | Age must be 18-100, etc. |
| 503 | Service unavailable | Model not loaded | Restart API server |
| 500 | Internal error | Unexpected exception | Check server logs |

### Example Error Response

```json
{
  "status_code": 422,
  "error": "Validation error",
  "details": [
    {
      "field": "Age",
      "message": "ensure this value is less than or equal to 100"
    }
  ]
}
```

---

## Performance & SLA

### Expected Response Times

| Endpoint | Typical Time | SLA |
|----------|--------------|-----|
| `/score` | 200-300ms | < 500ms |
| `/batch-score` (100 apps) | 5-10s | < 30s |
| `/explain` | 400-600ms | < 1s |

### Throughput

- Single endpoint: ~100-200 requests/second
- Batch endpoint: ~10-20 batches/second (100 apps each)

### Scaling

- For higher throughput, use multiple API workers:
  ```bash
  gunicorn app.api:app --workers 4 --threads 2
  ```

---

## Authentication (Future)

Currently: No authentication (development mode)

For production, add:
```python
# In app/api.py
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/score")
async def score(application: ApplicationRequest, credentials: HTTPAuthCredentials = Depends(security)):
    # Validate token...
    pass
```

---

## Rate Limiting (Future)

For production, use:
```bash
pip install slowapi

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/score")
@limiter.limit("100/minute")
async def score(application: ApplicationRequest):
    pass
```

---

## Logging & Monitoring

### Request/Response Logging

All requests logged with:
- Timestamp
- Endpoint
- Application features (anonymized)
- Response (score, decision)
- Response time
- Any errors

**Log Location:** `logs/api_requests.log`

### Example Log Entry

```
2026-04-26 10:30:45.123 | POST /score | Age=35 Income=65K CreditScore=750 | Score=0.28 Decision=APPROVE | Response Time=245ms
```

### Monitoring Metrics

- Request count per endpoint
- Average response time
- Error rate
- Model version in use
- Decision distribution (% APPROVE/REVIEW/REJECT)

---

## Testing the API

### Using cURL

```bash
# Test single scoring
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
```

### Using Python

```python
import requests

url = "http://localhost:8000/score"
payload = {
    "Age": 35,
    "Income": 65000,
    "CreditScore": 750,
    "LoanAmount": 200000,
    "YearsExperience": 10,
    "Gender": "M",
    "Education": "Bachelor",
    "City": "NewYork",
    "EmploymentType": "Employed"
}
response = requests.post(url, json=payload)
print(response.json())
```

### Using Swagger UI

1. Start API: `uvicorn app.api:app --reload`
2. Open: `http://localhost:8000/docs`
3. Click "Try it out" on any endpoint
4. Fill in parameters
5. Click "Execute"
