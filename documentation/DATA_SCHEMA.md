# Data Schema & Requirements

## Input Dataset Requirements

### File Format
- **CSV** (recommended) or **Parquet**
- Encoding: UTF-8
- Delimiter: comma (CSV)

### Required Columns

The system expects **exactly 10 columns** (as provided in the Kaggle dataset):

| Column Name | Data Type | Description | Constraints |
|-------------|-----------|-------------|-------------|
| `Age` | Integer | Applicant age in years | Should be 18-100 |
| `Income` | Float | Annual income in dollars | Should be > 0 |
| `CreditScore` | Integer | Credit score (typically 300-850) | Should be 0-1000 |
| `LoanAmount` | Float | Loan amount requested in dollars | Should be > 0 |
| `YearsExperience` | Integer | Years of employment experience | Should be >= 0 |
| `Gender` | String | Gender category | Values: M, F, Other |
| `Education` | String | Education level | Values: HighSchool, Bachelor, Master, PhD |
| `City` | String | City of residence | Any string |
| `EmploymentType` | String | Employment category | Values: Employed, Self-Employed, Unemployed, Retired |
| `LoanApproved` | Binary Integer | **TARGET VARIABLE** | Values: 0 (Rejected/Default), 1 (Approved) |

---

## Column Specifications

### Numerical Features

#### Age
- **Type:** Integer
- **Valid Range:** 18-100 years
- **Handling:** Flag values outside 18-100 as outliers
- **Usage:** Binning (age groups) in feature engineering

#### Income
- **Type:** Float
- **Valid Range:** > 0 (typically $20,000 - $500,000)
- **Missing Value Strategy:** Remove row or impute median
- **Usage:** Binning (income brackets) in feature engineering

#### CreditScore
- **Type:** Integer
- **Valid Range:** 0-1000 (typically 300-850)
- **Missing Value Strategy:** Impute median
- **Usage:** Binning (score tiers) in feature engineering

#### LoanAmount
- **Type:** Float
- **Valid Range:** > 0 (typically $5,000 - $500,000)
- **Missing Value Strategy:** Impute median
- **Usage:** Business metrics (expected loss calculation)

#### YearsExperience
- **Type:** Integer
- **Valid Range:** 0-80 years
- **Missing Value Strategy:** Impute 0 (no experience)
- **Usage:** Feature in modeling

---

### Categorical Features

#### Gender
- **Type:** String
- **Valid Values:** M, F, Other
- **Missing Value Strategy:** Impute mode
- **Encoding:** One-hot or target encoding

#### Education
- **Type:** String
- **Valid Values:** HighSchool, Bachelor, Master, PhD
- **Order:** HighSchool < Bachelor < Master < PhD (ordinal)
- **Missing Value Strategy:** Impute mode
- **Encoding:** Ordinal or target encoding

#### City
- **Type:** String
- **Valid Values:** Any city name
- **Cardinality:** May have many unique cities
- **Missing Value Strategy:** Impute mode
- **Encoding:** Target encoding (high cardinality categorical)

#### EmploymentType
- **Type:** String
- **Valid Values:** Employed, Self-Employed, Unemployed, Retired
- **Missing Value Strategy:** Impute mode
- **Encoding:** One-hot or target encoding

---

## Target Variable: LoanApproved

- **Type:** Binary integer (0 or 1)
- **Values:**
  - **1:** Loan approved (positive class, good outcome)
  - **0:** Loan rejected or applicant defaulted (negative class, risky outcome)
- **Class Distribution:** Check for imbalance
  - If < 5% are 1 → highly imbalanced
  - If 40-60% are 1 → balanced
  - Imbalance handling: SMOTE, class weights

---

## Data Quality Checks

### Missing Values

**Handling Strategy:**
1. **Numerical features:** Impute median per feature
2. **Categorical features:** Impute mode per feature
3. **Target variable:** Remove rows with missing target (can't label)
4. **Rows with > 30% missing:** Consider removing

**Logging:**
- Report % missing per column
- Log imputation actions

### Duplicates

**Handling:**
- Check for exact duplicate rows (should have none)
- Log and remove if found
- Report count removed

### Data Type Validation

**Expected Types:**
- Age, CreditScore, YearsExperience → Integer
- Income, LoanAmount → Float
- Gender, Education, City, EmploymentType → String (object)
- LoanApproved → Integer (0 or 1)

**Validation:**
- Raise error if types don't match
- Attempt coercion with warning if possible

### Outlier Detection

**Method:** IQR or Z-score

| Feature | Min | Max | Flag if outside |
|---------|-----|-----|------------------|
| Age | 18 | 100 | < 18 or > 100 |
| Income | 1 | 1,000,000 | < 1 or > 1,000,000 |
| CreditScore | 0 | 1000 | < 0 or > 1000 |
| LoanAmount | 1 | 500,000 | < 1 or > 500,000 |
| YearsExperience | 0 | 80 | < 0 or > 80 |

---

## Data Splits

### Train/Validation/Test Split

**Ratio:** 70% / 15% / 15%

**Strategy:** **Stratified split** (preserve target distribution)

```
Total: 100%
├─ Train: 70% (used for feature fitting + model training)
├─ Validation: 15% (used for hyperparameter tuning, model selection)
└─ Test: 15% (used for final evaluation, held out)
```

**Why Stratified?**
- If approval rate is 40% overall, each split has ~40% approval rate
- Prevents bias toward one class in smaller splits

---

## Example Dataset Format

```csv
Age,Income,CreditScore,LoanAmount,YearsExperience,Gender,Education,City,EmploymentType,LoanApproved
35,65000,750,200000,10,M,Bachelor,NewYork,Employed,1
28,45000,620,100000,5,F,HighSchool,LosAngeles,Employed,0
45,120000,800,350000,20,M,Master,Chicago,Self-Employed,1
23,30000,580,50000,2,F,Bachelor,Houston,Employed,0
```

---

## Data Preparation Checklist

Before running the pipeline:

- [ ] Dataset is in CSV format (UTF-8 encoded)
- [ ] Column names exactly match: Age, Income, CreditScore, LoanAmount, YearsExperience, Gender, Education, City, EmploymentType, LoanApproved
- [ ] All rows have valid target value (0 or 1)
- [ ] No confidential information (SSN, account numbers) in dataset
- [ ] File is saved in `data/raw/` directory

---

## Configuration in `feature_config.yaml`

The data schema is also defined in the feature config:

```yaml
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

missing_value_handling:
  strategy: "impute_median"  # or impute_mode, drop
  numerical_impute_with: "median"
  categorical_impute_with: "mode"

outlier_handling:
  method: "iqr"  # or zscore, clip
  iqr_multiplier: 1.5
```

---

## FAQ

**Q: What if my dataset has different column names?**
A: You must rename columns to match the expected names before running the pipeline. Or edit `feature_config.yaml` to specify custom column names (requires code changes).

**Q: What if I have missing values?**
A: The ingestion module will impute them (median for numerical, mode for categorical) and log the actions. You can configure the strategy in `feature_config.yaml`.

**Q: What if I have more/fewer features?**
A: This system is tailored for the 10 specified features. If you add features, update `feature_config.yaml`. If you remove features, features that depend on them will fail.

**Q: How many rows do I need?**
A: Minimum recommended: 1,000 rows. Ideally 10,000+ for reliable model training and evaluation.

**Q: What if my target variable is not binary?**
A: This system only supports binary classification (0 or 1). If you have multi-class, you must convert to binary or use a different system.
