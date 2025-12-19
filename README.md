# 🏦 AI-Powered Loan Risk Assessment System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![TCGM](https://img.shields.io/badge/tcgm-0.1.4-green.svg)](https://pypi.org/project/tcgm/0.1.4/)

A comprehensive machine learning system for assessing loan application risk and providing automated lending recommendations. The system uses Time-Cost Gradient Machine (TCGM) regression with asymmetric cost penalties to predict customer risk scores and recommend safe loan amounts.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Risk Assessment Logic](#risk-assessment-logic)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This system provides end-to-end loan risk assessment capabilities, from data analysis and model training to production deployment with real-time monitoring. It helps financial institutions make data-driven lending decisions while minimizing credit risk.

### Key Capabilities

- **Risk Prediction**: ML-powered risk scoring (0-1 scale) for loan applicants
- **Loan Recommendations**: Automated calculation of maximum safe loan amounts
- **Risk Tiering**: Classification into 5 risk tiers (A through E)
- **Interest Rate Adjustments**: Dynamic pricing based on risk assessment
- **Approval Decisions**: Automated recommendations (Approve, Manual Review, Decline)
- **Real-time Monitoring**: Drift detection and performance tracking
- **Web Interface**: User-friendly form for manual assessments

---

## ✨ Features

### 🤖 Machine Learning

- **TCGM Regressor** with asymmetric cost penalties (c_over=1.0, c_under=5.0)
- Penalizes underestimating risk 5x more than overestimating
- Trained on 30+ features including demographics, credit history, and financials
- Handles both numeric and categorical variables
- Achieves R² score of ~0.85 on test data

### 💰 Financial Analysis

- **Debt-to-Income (DTI) Ratio Analysis**: Maximum 43% DTI threshold
- **Credit Score Adjustments**: Multipliers based on credit worthiness (600-850)
- **Payment Capacity Calculation**: Monthly payment limits based on income
- **Conservative Lending Ratios**: Base multiplier of 30% of annual income
- **Risk-Adjusted Loan Amounts**: Automatically reduced for higher-risk applicants

### 🎨 User Interface

- **Beautiful Web Form**: Easy-to-use interface for entering customer data
- **Real-time Results**: Instant risk assessment and loan recommendations
- **Color-Coded Display**: Visual risk indicators (green to red)
- **Printable Reports**: Generate PDF-ready assessment reports
- **Responsive Design**: Works on desktop, tablet, and mobile

### 📊 Monitoring & Analytics

- **Database Logging**: All predictions stored with timestamps
- **Drift Detection**: Automated data distribution monitoring
- **Performance Metrics**: Track model accuracy over time
- **Grafana Dashboards**: Real-time visualization of predictions
- **Alerting System**: Notifications for drift or performance degradation

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                    (Web Form - Flask HTML)                      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FLASK APPLICATION                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Input      │→ │   Feature    │→ │   TCGM       │         │
│  │ Validation   │  │ Engineering  │  │   Model      │         │
│  └──────────────┘  └──────────────┘  └──────┬───────┘         │
│                                              │                  │
│  ┌──────────────┐  ┌──────────────┐  ◄──────┘                 │
│  │   Risk       │← │   Loan       │                            │
│  │  Tiering     │  │Recommendation│                            │
│  └──────┬───────┘  └──────────────┘                            │
└─────────┼───────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POSTGRESQL DATABASE                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ predictions  │  │drift_metrics │  │model_metrics │         │
│  │   (logs)     │  │  (monitoring)│  │(performance) │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING SYSTEM                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Evidently   │  │   Grafana    │  │   Alerts     │         │
│  │(Drift Check) │  │ (Dashboards) │  │  (Email)     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Risk Assessment Logic

### 1. Data Preprocessing

**Numeric Features** (25 features):
- Standardized using `StandardScaler`
- Includes: Age, credit score, income (log-transformed), debt ratios, payment history

**Categorical Features** (5 features):
- One-hot encoded using `OneHotEncoder`
- Includes: Employment status, education, marital status, home ownership, loan purpose

### 2. Risk Score Prediction

The TCGM model predicts a **Domain Risk Score** (0-1 scale):

- **0.0 - 0.3**: Low Risk (Tier A)
- **0.3 - 0.5**: Medium-Low Risk (Tier B)
- **0.5 - 0.65**: Medium Risk (Tier C)
- **0.65 - 0.8**: Medium-High Risk (Tier D)
- **0.8 - 1.0**: High Risk (Tier E)

**Asymmetric Cost Function**:
```
Cost = c_over × max(y_pred - y_true, 0) + c_under × max(y_true - y_pred, 0)

where:
  c_over = 1.0   (cost of overestimating risk)
  c_under = 5.0  (cost of underestimating risk)
```

This makes the model **conservative** - it prefers to be cautious rather than lend to risky customers.

### 3. Loan Amount Recommendation

The system calculates maximum safe loan amount using:

**Formula**:
```python
# Risk adjustment
risk_adjusted_multiplier = 0.3 × (1 - risk_score)

# Credit score adjustment
if credit_score >= 750: adjustment = 1.2
elif credit_score >= 700: adjustment = 1.1
elif credit_score >= 650: adjustment = 1.0
elif credit_score >= 600: adjustment = 0.8
else: adjustment = 0.6

# Payment capacity
monthly_income = annual_income / 12
max_monthly_debt = monthly_income × 0.43  # 43% DTI limit
available_payment = max_monthly_debt - existing_debt

# Loan from payment capacity (5% interest, 60 months)
max_loan_payment = available_payment × present_value_factor

# Loan from income multiple
max_loan_income = annual_income × risk_adjusted_multiplier × credit_adjustment

# Take minimum (most conservative)
recommended_loan = min(max_loan_payment, max_loan_income, annual_income × 0.5)
```

### 4. Interest Rate Adjustment

Base rate (5.0%) is adjusted based on risk tier:

| Risk Tier | Adjustment | Final Rate |
|-----------|------------|------------|
| A (Low)   | +0.0%      | 5.0%       |
| B (Med-Low)| +1.0%     | 6.0%       |
| C (Medium)| +2.0%      | 7.0%       |
| D (Med-High)| +3.5%    | 8.5%       |
| E (High)  | +5.0%      | 10.0%      |

### 5. Approval Decision Logic

```python
if risk_score < 0.3:
    decision = "AUTO_APPROVE"
elif risk_score < 0.5:
    decision = "APPROVE"
elif risk_score < 0.8:
    decision = "MANUAL_REVIEW"
else:
    decision = "DECLINE"
```

---

## 🛠️ Technology Stack

### Machine Learning
- **scikit-learn** 1.5.2 - Preprocessing pipelines
- **TCGM** 0.1.4 - Time-Cost Gradient Machine regressor
- **pandas** 2.2.0 - Data manipulation
- **numpy** 1.26.3 - Numerical computations

### Web Framework
- **Flask** 3.0.0 - Web application framework
- **Gunicorn** 21.2.0 - Production WSGI server
- **HTML/CSS** - Custom responsive UI

### Database
- **PostgreSQL** 15+ - Production database
- **psycopg2-binary** 2.9.9 - PostgreSQL adapter

### Monitoring
- **Evidently** 0.4.12 - Drift detection
- **Grafana** 10+ - Visualization dashboards
- **MLflow** 2.10.0 - Model tracking and versioning

### Deployment
- **Render** - Cloud hosting platform
- **Git/GitHub** - Version control
- **pgAdmin4** - Database management

---


## 🚀 Usage

### Web Interface

1. Navigate to the home page
2. Fill in customer information:
   - Personal details (age, employment)
   - Credit information (score, payment history)
   - Loan details (duration, purpose)
   - Financial information (income, assets)
3. Click "Assess Risk & Get Decision"
4. View results:
   - Risk score and tier
   - Approval decision
   - Recommended loan amount
   - Interest rates
   - Monthly payment capacity

### API Endpoints

#### Health Check
```bash
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_version": "1.0",
  "database": "connected",
  "timestamp": "2025-12-19T10:30:00"
}
```

#### Risk Assessment
```bash
POST /assess
Content-Type: application/x-www-form-urlencoded

# Form data with all customer fields
```

---

## 📊 Model Details

### Training Data

- **Dataset**: Financial Risk for Loan Approval (Kaggle)
- **Samples**: ~10,000 loan applications
- **Features**: 30+ variables
- **Target**: Domain Risk Score (0-1)
- **Split**: 80% training, 20% testing

### Model Performance

| Metric | Value |
|--------|-------|
| MAE | 0.0829 |
| RMSE | 0.1019 |
| R² Score | ~0.85 |
| Asymmetric MAE | 0.2587 |

### Feature Importance

Top 10 most important features:
1. Credit Score (23.5%)
2. Annual Income (18.2%)
3. Payment History (12.8%)
4. Total Debt-to-Income Ratio (9.4%)
5. Length of Credit History (7.6%)
6. Number of Open Credit Lines (6.3%)
7. Previous Loan Defaults (5.9%)
8. Employment Status (4.8%)
9. Loan Duration (4.2%)
10. Age (3.7%)


**Live URL**: 


---

## 📈 Monitoring

### Database Logging

Every prediction is automatically logged with:
- Customer ID
- Input features
- Risk score
- Recommended loan amount
- Approval decision
- Timestamp

### Drift Detection

Run periodically (via cron):
```bash
python drift_monitor.py
```

Monitors:
- Feature distribution changes
- Concept drift
- Population shift

### Performance Tracking

Track model performance over time:
```bash
python monitoring/alerts.py
```

Alert conditions:
- Drift > 50%
- High risk rate > 70%
- Performance degradation

### Grafana Dashboards

Access at: `http://localhost:3000`

Dashboards include:
- Predictions over time
- Risk score distribution
- Approval decision breakdown
- Drift metrics timeline

---

## 📖 API Documentation

### POST /assess

Submit customer data for risk assessment.

**Parameters** (all required):

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| customer_id | string | Unique customer identifier | "CUST001" |
| Age | integer | Customer age (18-100) | 35 |
| CreditScore | integer | Credit score (300-850) | 680 |
| AnnualIncome_log | float | Log of annual income | 11.0 |
| ... | ... | (30+ more parameters) | ... |

**Response**:
```json
{
  "customer_id": "CUST001",
  "risk_assessment": {
    "risk_score": 0.4523,
    "risk_tier": "Medium-Low Risk",
    "risk_tier_code": "B",
    "description": "Good credit profile"
  },
  "loan_recommendation": {
    "max_approved_amount": 45234.56,
    "monthly_payment_capacity": 1234.50,
    "estimated_monthly_payment": 1111.05
  },
  "lending_terms": {
    "base_interest_rate": 5.0,
    "risk_adjusted_rate": 6.0,
    "approval_decision": "APPROVE"
  },
  "metadata": {
    "model_version": "1.0",
    "prediction_timestamp": "2025-12-19T10:30:00",
    "annual_income": 60000.00,
    "credit_score": 680
  }
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Ensure all tests pass
- 

## 👥 Author

- **Chidiebere V. Christopher** - *Data Scientist || Machine Learning Researcher* - [MyGitHub](https://github.com/93Chidiebere)

---

## 🙏 Acknowledgments

- Dataset: [Financial Risk for Loan Approval - Kaggle](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval/data?select=Loan.csv)
- TCGM Library: [Time-Cost Gradient Machine implementation](https://pypi.org/project/tcgm/0.1.4/)
- Flask Framework: Web application framework
- Render: Cloud hosting platform

---

**Project Link**: [https://github.com/93Chidiebere/LoanRegression](https://github.com/93Chidiebere/LoanRegression)

**Live Demo**: [https://your-app-name.onrender.com](https://loan-risk-assessment.onrender.com)

**Email**: vchidiebere.vc@gmail.com

---

## 📚 Additional Resources

- [Complete Deployment Guide](docs/DEPLOYMENT.md)
- [Model Training Notebook](notebooks/RiskScoreModel.ipynb)

---

## 🎯 Roadmap

- [x] Basic risk assessment model
- [x] Web interface
- [x] Database logging
- [x] Drift detection
- [ ] A/B testing framework
- [ ] Mobile app
- [ ] Multi-language support
- [ ] Advanced explainability (SHAP values)
- [ ] Batch processing API
- [ ] Real-time fraud detection

---

**⭐ If you find this project useful, please consider giving it a star!**

**📢 Share this project with others who might benefit from it!**

---

*Last Updated: December 2025*
