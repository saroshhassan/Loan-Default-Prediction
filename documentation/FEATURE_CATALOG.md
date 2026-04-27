# Feature Catalog

## Overview

This document lists all engineered features created by the `feature_engineering` module. Each feature is designed to capture loan default risk signals from raw applicant data.

---

## Raw Features (From Dataset)

| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `Age` | Numerical | Raw | Applicant age in years |
| `Income` | Numerical | Raw | Annual income in dollars |
| `CreditScore` | Numerical | Raw | Credit history score |
| `LoanAmount` | Numerical | Raw | Requested loan amount |
| `YearsExperience` | Numerical | Raw | Years of employment |
| `Gender` | Categorical | Raw | M, F, Other |
| `Education` | Categorical | Raw | HighSchool, Bachelor, Master, PhD |
| `City` | Categorical | Raw | City of residence |
| `EmploymentType` | Categorical | Raw | Employed, Self-Employed, Unemployed, Retired |

---

## Engineered Features

### 1. Binned Age Groups

| Feature Name | Source | Computation | Data Type | Why Relevant |
|--------------|--------|-------------|-----------|--------------|
| `age_group_18_25` | Age | 1 if 18 ≤ Age < 25, else 0 | Binary | Younger applicants may have higher risk |
| `age_group_25_35` | Age | 1 if 25 ≤ Age < 35, else 0 | Binary | Prime working years |
| `age_group_35_50` | Age | 1 if 35 ≤ Age < 50, else 0 | Binary | Stable income period |
| `age_group_50_plus` | Age | 1 if Age ≥ 50, else 0 | Binary | Approaching retirement |

**Encoding Strategy:** One-hot (drop first to avoid multicollinearity)

---

### 2. Income Brackets

| Feature Name | Source | Computation | Data Type | Why Relevant |
|--------------|--------|-------------|-----------|--------------|
| `income_low` | Income | 1 if Income < $35,000, else 0 | Binary | Lower income = higher default risk |
| `income_mid` | Income | 1 if $35,000 ≤ Income < $75,000, else 0 | Binary | Middle income tier |
| `income_high` | Income | 1 if Income ≥ $75,000, else 0 | Binary | Higher income = lower risk |

**Encoding Strategy:** One-hot (drop first)

---

### 3. Credit Score Tiers

| Feature Name | Source | Computation | Data Type | Why Relevant |
|--------------|--------|-------------|-----------|--------------|
| `credit_poor` | CreditScore | 1 if CreditScore < 600, else 0 | Binary | Poor credit history = high risk |
| `credit_fair` | CreditScore | 1 if 600 ≤ CreditScore < 700, else 0 | Binary | Fair credit |
| `credit_good` | CreditScore | 1 if 700 ≤ CreditScore < 800, else 0 | Binary | Good credit |
| `credit_excellent` | CreditScore | 1 if CreditScore ≥ 800, else 0 | Binary | Excellent credit = low risk |

**Encoding Strategy:** One-hot (drop first)

---

### 4. Debt-to-Income Ratio

| Feature Name | Source | Computation | Data Type | Why Relevant |
|--------------|--------|-------------|-----------|--------------|
| `debt_to_income` | LoanAmount, Income | LoanAmount / Income | Float (0-1+) | Core credit metric; higher = more stress |

**Encoding Strategy:** StandardScaler normalization

**Business Context:** DTI > 0.5 often signals difficulty servicing debt

---

### 5. Experience vs Age Proxy

| Feature Name | Source | Computation | Data Type | Why Relevant |
|--------------|--------|-------------|-----------|--------------|
| `exp_to_age_ratio` | YearsExperience, Age | YearsExperience / (Age - 18) | Float | Work history relative to adult age |

**Encoding Strategy:** StandardScaler normalization

**Business Context:** High ratio = consistent employment; low ratio = job hopping or late career entry

---

### 6. Loan Amount Categories

| Feature Name | Source | Computation | Data Type | Why Relevant |
|--------------|--------|-------------|-----------|--------------|
| `loan_small` | LoanAmount | 1 if LoanAmount < $100,000, else 0 | Binary | Smaller loans easier to service |
| `loan_medium` | LoanAmount | 1 if $100,000 ≤ LoanAmount < $300,000, else 0 | Binary | Medium-sized loans |
| `loan_large` | LoanAmount | 1 if LoanAmount ≥ $300,000, else 0 | Binary | Large loans = higher risk |

**Encoding Strategy:** One-hot (drop first)

---

### 7. Employment Type Features (Encoded)

| Feature Name | Source | Encoding | Data Type | Why Relevant |
|--------------|--------|----------|-----------|--------------|
| `emp_employed` | EmploymentType | 1 if Employed, else 0 | Binary | Stable employment = lower risk |
| `emp_self_employed` | EmploymentType | 1 if Self-Employed, else 0 | Binary | Variable income = higher risk |
| `emp_unemployed` | EmploymentType | 1 if Unemployed, else 0 | Binary | No income = very high risk |
| `emp_retired` | EmploymentType | 1 if Retired, else 0 | Binary | Fixed income, predictable |

**Encoding Strategy:** One-hot (drop first)

---

### 8. Education Features (Encoded)

| Feature Name | Source | Encoding | Data Type | Why Relevant |
|--------------|--------|----------|-----------|--------------|
| `edu_highschool` | Education | 1 if HighSchool, else 0 | Binary | Lower earning potential |
| `edu_bachelor` | Education | 1 if Bachelor, else 0 | Binary | Standard college degree |
| `edu_master` | Education | 1 if Master, else 0 | Binary | Advanced degree |
| `edu_phd` | Education | 1 if PhD, else 0 | Binary | Highest education = lower risk |

**Encoding Strategy:** One-hot (drop first) — could also use ordinal (0, 1, 2, 3)

---

### 9. Gender Features (Encoded)

| Feature Name | Source | Encoding | Data Type | Why Relevant |
|--------------|--------|----------|-----------|--------------|
| `gender_m` | Gender | 1 if Male, else 0 | Binary | For demographic analysis |
| `gender_f` | Gender | 1 if Female, else 0 | Binary | For demographic analysis |
| `gender_other` | Gender | 1 if Other, else 0 | Binary | For demographic analysis |

**Encoding Strategy:** One-hot (drop first)

---

### 10. City Features (Target Encoding)

| Feature Name | Source | Encoding | Data Type | Why Relevant |
|--------------|--------|----------|-----------|--------------|
| `city_encoded` | City | Target encoding: mean(default_rate) per city | Float | Geographic default risk patterns |

**Encoding Strategy:** Target encoding (fit on train set only)

**Example:**
- NYC has 5% default rate → city_encoded = 0.05 for NYC applicants
- LA has 8% default rate → city_encoded = 0.08 for LA applicants

---

## Feature Engineering Pipeline

### Step 1: Data Validation
- Check for missing values
- Handle outliers (flag or cap)

### Step 2: Binning & Categorization
- Create age, income, credit score, loan amount bins
- Result: ~12 new binary features

### Step 3: Ratio Features
- Compute DTI, experience-to-age ratio
- Result: 2 numerical features

### Step 4: Encoding Categorical
- One-hot encode employment, education, gender
- Target encode city
- Result: ~8 encoded features

### Step 5: Feature Selection
- Compute feature importance (tree-based or univariate)
- Keep top-20 features by importance
- Drop features with < 0.01 importance threshold

### Step 6: Scaling
- StandardScaler on numerical features
- Leave binary features unscaled

---

## Final Feature Set (Typical)

After feature engineering, expect **~25-30 features** including:
- 12 binned categorical features (age, income, credit score, loan amount)
- 2 ratio features (DTI, exp-to-age)
- 4 employment features
- 4 education features
- 3 gender features
- 1 city encoding
- Plus original continuous features (scaled)

---

## Feature Importance Interpretation

**Why each feature matters for default prediction:**

| Feature | Importance | Why |
|---------|----------|-----|
| `credit_poor` | Very High | Past credit behavior is strong default predictor |
| `debt_to_income` | Very High | DTI > 0.5 indicates payment stress |
| `emp_unemployed` | High | No income = high default risk |
| `age_group_18_25` | High | Younger applicants often higher risk |
| `income_low` | High | Lower income = less buffer for payments |
| `loan_large` | Medium | Larger payments harder to sustain |
| `education` | Medium | Education correlates with income stability |
| `city_encoded` | Low-Medium | Geographic patterns exist but weaker signal |

---

## Feature Configuration (feature_config.yaml)

Features are configured in `feature_config.yaml`:

```yaml
features:
  binning:
    age:
      bins: [18, 25, 35, 50, 100]
      labels: ["18-25", "25-35", "35-50", "50+"]
    income:
      bins: [0, 35000, 75000, 1000000]
      labels: ["low", "mid", "high"]
    credit_score:
      bins: [0, 600, 700, 800, 1000]
      labels: ["poor", "fair", "good", "excellent"]

  encoding:
    strategy: "one_hot"  # or target_encoding
    categorical_features:
      - Gender
      - Education
      - City
      - EmploymentType

  selection:
    method: "importance"  # or correlation
    keep_top_k: 25
    importance_threshold: 0.01
```

---

## Preventing Data Leakage

**Critical:** All feature transformations (encoding, binning, scaling) must be fit on **train set only**, then applied to val/test:

```python
# Correct: fit on train, transform all
encoder.fit(X_train, y_train)
X_train_encoded = encoder.transform(X_train)
X_val_encoded = encoder.transform(X_val)  # No refitting!
X_test_encoded = encoder.transform(X_test)

# Wrong: refitting on val/test causes leakage
encoder.fit(X_val)  # ← Data leakage!
```

---

## Feature Update Workflow

To add new features:

1. Edit `feature_config.yaml` to define the new feature
2. Add computation logic in `src/feature_engineering/`
3. Update `FeatureEngineeringPipeline` class
4. Retrain models
5. Re-evaluate with new feature importance

Example: Adding "loan_to_income_ratio":
```yaml
features:
  ratios:
    loan_to_income:
      numerator: "LoanAmount"
      denominator: "Income"
      threshold: 5  # Flag if > 5x income
```

Then in code:
```python
def create_loan_to_income(df):
    return df['LoanAmount'] / df['Income']
```
