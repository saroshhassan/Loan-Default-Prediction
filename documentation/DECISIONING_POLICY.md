# Decisioning Policy & Business Simulation

## Core Concept

The decisioning engine converts a **risk score** (probability of default) into a **business decision** and simulates the financial impact.

```
Risk Score (P(Default)) 
    ↓
    Decision Thresholds
    ↓
Decision (APPROVE / REVIEW / REJECT)
    ↓
Business Metrics (Approval Rate, Default Rate, Expected Loss)
```

---

## Risk Score Interpretation

**Risk Score Range:** [0, 1]

| Score | Interpretation | Example |
|-------|-----------------|---------|
| 0.05 | Very low risk (5% chance of default) | Young, employed, high credit score |
| 0.20 | Low risk (20% chance of default) | Stable employment, good credit |
| 0.50 | Medium risk (50% chance of default) | Some financial stress signals |
| 0.80 | High risk (80% chance of default) | Poor credit, unemployment |
| 0.95 | Very high risk (95% chance of default) | Bankruptcy history, high debt |

---

## Decision Thresholds

The system defines **two critical thresholds** that divide the risk spectrum:

```
Risk Score: [0 ————— approve_th ————— reject_th ————— 1]
Decision:   [APPROVE ———— REVIEW ———————— REJECT ————]
```

### Approve Threshold

**Definition:** Risk score below this → APPROVE (lend)

**Default:** 0.30 (30% default probability)

**Interpretation:** "Applicants with < 30% default risk are safe to approve"

**Business Impact:**
- Lower threshold = More approvals, higher portfolio default rate
- Higher threshold = Fewer approvals, lower portfolio default rate

**Tuning:**
- Increase if portfolio default rate is too high
- Decrease if approval rate is too low

### Reject Threshold

**Definition:** Risk score above this → REJECT (decline loan)

**Default:** 0.70 (70% default probability)

**Interpretation:** "Applicants with > 70% default risk are too risky to approve"

**Business Impact:**
- Lower threshold = Fewer rejections (more approvals but riskier)
- Higher threshold = More rejections (fewer approvals but safer)

**Tuning:**
- Increase if wanting to reject obvious bad loans only
- Decrease if wanting to be conservative

### Review Threshold Range

**Definition:** Risk score between approve_th and reject_th → REVIEW (manual decision)

**Range:** [0.30, 0.70]

**Interpretation:** "Marginal applicants (30-70% default risk) get manual review"

**Business Logic:**
- Analyst reviews all metrics (income, employment, credit history)
- May approve with conditions (higher interest rate, co-signer)
- May reject if other concerns present

---

## Decision Policy Definition

**Location:** `decision_config.yaml`

```yaml
decision_policy:
  approve_threshold: 0.30
  reject_threshold: 0.70
  review_required: true  # Flag marginal applications
  
  # Optional: Risk-based pricing
  risk_pricing:
    enabled: false
    score_0_25:
      interest_rate: 4.5
    score_25_50:
      interest_rate: 5.5
    score_50_75:
      interest_rate: 7.5
```

---

## Business Metrics & Simulation

### Key Metrics Computed

#### 1. Approval Rate

**Definition:** % of applicants approved

```
Approval Rate = (# Approved) / (# Total) × 100
```

**Example:** 75% → 3 out of 4 applicants are approved

**Business Impact:**
- Higher approval rate → More revenue, higher default losses
- Lower approval rate → Less revenue, lower default losses

#### 2. Default Rate (In Approved Portfolio)

**Definition:** % of approved applicants who actually default

```
Default Rate = (# Defaults among Approved) / (# Approved) × 100
```

**Example:** 12% → 1 in 8 approved applicants default

**Business Impact:**
- Higher default rate → More credit losses
- Lower default rate → Healthier portfolio

#### 3. Expected Loss (Simulated)

**Definition:** Total financial loss from defaults in approved portfolio

**Formula:**
```
Expected Loss = Σ(Loan Amount × P(Default)) for all approved applicants
```

**Example:**
```
Applicant 1: Loan $200K, Risk Score 0.2 → Loss = 200K × 0.2 = $40K
Applicant 2: Loan $100K, Risk Score 0.4 → Loss = 100K × 0.4 = $40K
Total Expected Loss = $80K
```

**Business Impact:**
- Higher expected loss → Reserves needed for defaults
- Lower expected loss → Better portfolio quality

#### 4. False Positive Rate (Wrongly Rejected)

**Definition:** % of actually good applicants who were rejected

```
False Positive Rate = (# Good applicants rejected) / (# Good applicants total) × 100
```

**Example:** 5% → 1 in 20 good applicants incorrectly rejected

**Business Impact:**
- Missed revenue opportunity (should have approved)
- Customer dissatisfaction

#### 5. Precision of Fraud Detection

**Definition:** Of rejected applicants, % who would actually default

```
Precision = (# Would-be defaults rejected) / (# Total rejected) × 100
```

**Example:** 85% → 85% of rejected applicants truly would have defaulted

**Business Impact:**
- Higher precision → Fewer false negatives (good targeting)
- Lower precision → Many false positives (over-rejecting)

#### 6. Recall of Fraud Detection

**Definition:** Of actual defaults, % successfully rejected

```
Recall = (# Defaults rejected) / (# Actual defaults) × 100
```

**Example:** 75% → Caught 3 out of 4 true defaults

**Business Impact:**
- Higher recall → Better risk mitigation
- Lower recall → More defaults slip through

---

## Threshold Optimization Workflow

The system sweeps across all possible threshold pairs and identifies **optimal thresholds** based on business objectives.

### Step 1: Define Objective

Choose what to optimize for:

```
Option A: Maximize Approval Rate (while keeping Default Rate < 15%)
Option B: Minimize Expected Loss (while keeping Approval Rate > 70%)
Option C: Balance Approval Rate & Default Rate (Pareto frontier)
```

### Step 2: Sweep Thresholds

```python
approval_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, ...]
reject_thresholds = [0.50, 0.55, 0.60, ..., 0.95]

results = []
for app_th in approval_thresholds:
    for rej_th in reject_thresholds:
        if app_th < rej_th:  # Valid pair
            metrics = simulate_business_impact(
                risk_scores, 
                app_th, 
                rej_th
            )
            results.append(metrics)
```

### Step 3: Identify Optimal Thresholds

```
Best for Approval:
├─ Thresholds: (0.35, 0.75)
├─ Approval Rate: 85%
├─ Default Rate: 16%
└─ Expected Loss: $250K

Best for Safety:
├─ Thresholds: (0.25, 0.60)
├─ Approval Rate: 60%
├─ Default Rate: 8%
└─ Expected Loss: $120K
```

### Step 4: Output Tradeoff Curves

```
Approval Rate vs Default Rate:
├─ As approval rate increases, default rate increases
├─ Pareto frontier shows best combinations
└─ Business selects target based on risk appetite
```

---

## Tradeoff Analysis

### Approval Rate vs Default Rate

**Question:** How many defaults do we accept to get more approvals?

```
Thresholds       Approval Rate   Default Rate   Expected Loss
(0.20, 0.80)     90%             18%            $280K
(0.25, 0.70)     78%             12%            $180K
(0.30, 0.65)     65%             8%             $120K
(0.35, 0.60)     52%             5%             $80K
```

**Tradeoff:** Going from 78% approval to 65% approval avoids $60K in losses

---

### Approval Rate vs False Positive Rate

**Question:** How many good applicants do we wrongly reject to be more selective?

```
Thresholds       Approval Rate   False Pos Rate   Wrongly Rejected
(0.30, 0.70)     78%             5%              ~4 out of 80 good
(0.25, 0.65)     68%             3%              ~2 out of 70 good
```

**Insight:** Tighter thresholds reduce false positives (fewer missed opportunities)

---

## Risk Segmentation

The system can also segment applicants into risk tiers:

```
Risk Segment       Score Range    # Applicants   Default Rate   Approval Decision
Low Risk           0.00 - 0.25    2,000          3%             Auto-approve
Moderate Risk      0.25 - 0.50    3,000          12%            Manual review
High Risk          0.50 - 0.75    2,500          35%            Conditional approval
Very High Risk     0.75 - 1.00    500            75%            Auto-reject
```

**Business Use:**
- Low risk → Instant decision, no review
- Moderate risk → Route to analyst
- High risk → Decline or offer at premium rate
- Very high risk → Immediate decline

---

## Configuration Example

```yaml
# decision_config.yaml

decision_policy:
  # Default thresholds
  approve_threshold: 0.30
  reject_threshold: 0.70
  
  # Business rules
  review_threshold_range: true  # Route marginal to analyst
  
  # Risk segments
  risk_segments:
    low:
      min: 0.00
      max: 0.25
      action: "auto_approve"
      interest_rate: 4.5
    moderate:
      min: 0.25
      max: 0.50
      action: "manual_review"
      interest_rate: 5.5
    high:
      min: 0.50
      max: 0.75
      action: "conditional_approval"
      interest_rate: 7.5
    very_high:
      min: 0.75
      max: 1.00
      action: "auto_reject"
      interest_rate: null
  
  # Business constraints
  constraints:
    min_approval_rate: 0.60  # Keep at least 60% approval
    max_default_rate: 0.15   # Don't exceed 15% default rate
    max_expected_loss: 300000  # Cap expected loss at $300K

# Optimization objective
optimization:
  objective: "balance"  # or "maximize_approval", "minimize_loss"
```

---

## Expected Loss Calculation (Deep Dive)

### Simple Example

```
Portfolio of 5 applicants:
├─ Applicant A: $200K loan, 10% default risk → Expected loss = $20K
├─ Applicant B: $150K loan, 20% default risk → Expected loss = $30K
├─ Applicant C: $300K loan, 5% default risk → Expected loss = $15K
├─ Applicant D: $100K loan, 30% default risk → Expected loss = $30K
└─ Applicant E: $250K loan, 15% default risk → Expected loss = $37.5K

Total Expected Loss = $20K + $30K + $15K + $30K + $37.5K = $132.5K
```

**Interpretation:** Over many portfolios like this, we expect ~$132.5K in losses due to defaults.

### Impact on Reserve Adequacy

If regulators require reserves equal to expected loss:
- Portfolio with $132.5K expected loss → Need $132.5K in capital reserves
- Tighter thresholds (lower expected loss) → Lower capital requirements

---

## Monitoring & Adjustment

### After Deployment

Monitor actual outcomes vs. predictions:

```
1. Track actual default rate vs. predicted
   → If actual > predicted: Model is too optimistic, tighten thresholds
   → If actual < predicted: Model is conservative, can loosen thresholds

2. Monitor approval rate vs. target
   → If too low: Loosen thresholds to increase approvals
   → If too high: Tighten thresholds to maintain quality

3. Monitor expected loss vs. budget
   → If expected loss increasing: Tighten thresholds
```

---

## FAQ

**Q: How do I choose between 0.30 and 0.35 for approve_threshold?**

A: Run tradeoff analysis (in this documentation), see impact on approval/default rates, choose based on business goals.

**Q: Can thresholds be adjusted after model deployment?**

A: Yes! Thresholds are independent of model. Retrain model quarterly, adjust thresholds monthly based on business performance.

**Q: What if I want different thresholds for different loan amounts?**

A: Add risk-based thresholds in config:
```yaml
risk_thresholds:
  small_loans:  # < $100K
    approve: 0.40
    reject: 0.75
  large_loans:  # > $300K
    approve: 0.25
    reject: 0.65
```

**Q: How does this handle protected attributes (race, gender, age)?**

A: This system treats attributes as regular features. For compliance, separately audit: "Are approval rates different by demographic?" If yes, adjust policy or remove features.
