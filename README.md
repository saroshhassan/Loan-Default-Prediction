# Risk Assessment Manager - Loan Default Prediction System

A production-grade machine learning system that predicts loan approval risk and supports lending decisions. This system bridges prediction analytics with actionable business decisions through calibrated risk scoring and automated decision policies.

## Features

- **End-to-end ML Pipeline:** Data ingestion → Feature Engineering → Model Training → Risk Scoring
- **Multiple Models:** Baseline Logistic Regression & Advanced Ensemble Methods (XGBoost, LightGBM)
- **Explainability:** SHAP-based model interpretability for regulatory compliance
- **Decision Engine:** Automated approval/review/reject policies based on risk thresholds
- **API & Dashboard:** FastAPI backend with Streamlit frontend for serving predictions
- **Comprehensive Monitoring:** Cross-validation, financial impact evaluation, and performance reporting

## Project Structure

```
RiskAssessmentManager/
├── data/                 # Raw and processed datasets
│   ├── raw/             # Kaggle loan dataset
│   └── processed/       # Cleaned, feature-engineered data
├── src/
│   ├── data_ingestion/  # CSV loading and validation
│   ├── feature_engineering/ # Data preprocessing and feature creation
│   ├── modeling/        # Model training and selection
│   ├── inference/       # Prediction pipeline
│   ├── decisioning/     # Decision policies
│   ├── evaluation/      # Model evaluation metrics
│   └── utils/           # Configuration management
├── configs/             # YAML configuration files
├── notebooks/           # Jupyter notebooks for exploration
├── models/              # Trained model artifacts
├── documentation/       # Comprehensive guides
└── tests/               # Unit and integration tests
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda
- Kaggle API credentials (optional, for automatic data download)
- ~2GB disk space for models and data

### Installation

1. **Clone/Navigate to the project:**
   ```bash
   cd d:/My\ Projects/RiskAssessmentManager
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate (Windows)
   .venv\Scripts\activate
   
   # Activate (macOS/Linux)
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set up configuration (Optional):**
   ```bash
   # Copy and customize if needed
   cp configs/config.yaml.example configs/config.yaml
   ```

### Running the Training Pipeline

```bash
# Execute the full training pipeline
python train.py
```

This will:
- Load and validate the loan dataset
- Perform exploratory data analysis
- Engineer features
- Train baseline and advanced models
- Evaluate performance metrics
- Generate decision policies
- Save trained models and reports

### Serving Predictions

**FastAPI Server:**
```bash
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```
Access API docs at `http://localhost:8000/docs`

**Streamlit Dashboard:**
```bash
streamlit run app/dashboard.py
```
Access dashboard at `http://localhost:8501`

## Configuration

All system configurations are managed through YAML files in the `configs/` directory:

- **config.yaml** - Main system settings (data paths, feature engineering)
- **model_config.yaml** - Model hyperparameters and training settings
- **feature_config.yaml** - Feature definitions and engineering rules

See [CONFIGURATION_GUIDE.md](documentation/CONFIGURATION_GUIDE.md) for detailed options.

## Data Requirements

The system expects a loan dataset with:
- **Applicant features:** Income, employment history, credit score, loan amount, etc.
- **Outcome variable:** Loan approval status (binary: approved/rejected)
- **Format:** CSV file with headers

Raw data should be placed in `data/raw/` directory.

## Key Models

| Model | Type | Speed | Accuracy | Interpretability |
|-------|------|-------|----------|------------------|
| Logistic Regression | Baseline | Very Fast | Good (0.65-0.75 AUC) | Excellent |
| XGBoost | Advanced | Fast | Excellent (0.80+ AUC) | Good |
| LightGBM | Advanced | Very Fast | Excellent (0.80+ AUC) | Good |

## Documentation

Comprehensive guides are available in the `documentation/` folder:

- [ARCHITECTURE.md](documentation/ARCHITECTURE.md) - System design and data flow
- [MODEL_GUIDE.md](documentation/MODEL_GUIDE.md) - Model training and selection
- [FEATURE_CATALOG.md](documentation/FEATURE_CATALOG.md) - Feature definitions
- [DECISIONING_POLICY.md](documentation/DECISIONING_POLICY.md) - Decision rules
- [API_REFERENCE.md](documentation/API_REFERENCE.md) - REST API documentation
- [DEPLOYMENT.md](documentation/DEPLOYMENT.md) - Production deployment guide
- [TROUBLESHOOTING.md](documentation/TROUBLESHOOTING.md) - Common issues and solutions

## Testing

Run the test suite:
```bash
pytest tests/
pytest --cov=src tests/  # With coverage report
```

## Dependencies

Key Python packages:
- **Data Processing:** pandas, numpy, scikit-learn
- **Modeling:** xgboost, lightgbm, optuna
- **API:** fastapi, uvicorn
- **Dashboard:** streamlit
- **Explainability:** shap
- **Development:** pytest, black, flake8, mypy

See [requirements.txt](requirements.txt) for the complete list.

## License

See [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or contributions, refer to [TROUBLESHOOTING.md](documentation/TROUBLESHOOTING.md) or check existing documentation.

---

**Last Updated:** May 2026
