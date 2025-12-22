"""
FastAPI Web Application for Loan Risk Assessment
• User enters NAIRA values
• Model internally uses LOG values
• Credit-report-driven behavioral governance
• No database dependency
• Render deployment ready
"""

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def safe_log(x: float) -> float:
    return float(np.log(max(x, 1.0)))

def safe_exp(x: float) -> float:
    return float(np.exp(x))

# --------------------------------------------------
# Behavioral Risk Index (NEW – GOVERNANCE LAYER)
# --------------------------------------------------

def compute_behavioral_risk_index(
    max_dpd,
    dpd_30,
    dpd_60,
    dpd_90,
    months_since_default,
    restructures
):
    score = 0.0
    score += min(max_dpd / 90, 1.0) * 0.30
    score += min(dpd_30 / 5, 1.0) * 0.15
    score += min(dpd_60 / 3, 1.0) * 0.20
    score += min(dpd_90 / 2, 1.0) * 0.25
    score += (1 - min(months_since_default / 24, 1.0)) * 0.20
    score += min(restructures / 3, 1.0) * 0.10
    return min(score, 1.0)

# --------------------------------------------------
# App & Model
# --------------------------------------------------

app = FastAPI(title="Loan Risk Assessment System")

with open("risk_assessment_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

FEATURE_NAMES = [
    "Age", "Experience", "JobTenure", "CreditScore", "PaymentHistory",
    "LengthOfCreditHistory", "NumberOfOpenCreditLines",
    "NumberOfCreditInquiries",
    "LoanDuration", "BaseInterestRate", "InterestRate",
    "TotalDebtToIncomeRatio",
    "MonthlyIncome_log", "AnnualIncome_log",
    "MonthlyLoanPayment_log", "LoanAmount_log",
    "MonthlyDebtPayments_log",
    "EmploymentStatus", "EducationLevel",
    "MaritalStatus", "HomeOwnershipStatus", "LoanPurpose"
]

# --------------------------------------------------
# UI – HOME
# --------------------------------------------------

HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Loan Risk Assessment</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body { font-family: Segoe UI; background: #f2f4f8; padding: 20px; }
.container { max-width: 1100px; margin:auto; background:white; padding:35px; border-radius:14px; }
.section { margin-bottom:30px; }
.section h3 { color:#4f46e5; border-bottom:2px solid #4f46e5; padding-bottom:6px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; }
input,select { padding:10px; border-radius:6px; border:1px solid #ccc; }
button { width:100%; padding:16px; background:#4f46e5; color:white; border:none; border-radius:40px; font-size:1.1em; }
.note { background:#eef2ff; padding:12px; border-left:4px solid #4f46e5; margin-bottom:25px; }
</style>
</head>

<body>
<div class="container">
<h1>Loan Risk Assessment</h1>
<p style="text-align:center;color:#666">Model-driven risk, behavior-driven governance</p>

<div class="note"><b>All financial values are in ₦ (Naira)</b></div>

<form method="post" action="/assess">

<div class="section">
<h3>👤 Customer Profile</h3>
<div class="grid">
<input name="customer_id" placeholder="Customer ID" required>
<input name="Age" type="number" placeholder="Age" required>
<input name="Experience" type="number" placeholder="Years of Experience">
<input name="JobTenure" type="number" placeholder="Job Tenure">
</div>
</div>

<div class="section">
<h3>💳 Credit Report (Behavioural)</h3>
<div class="grid">
<input name="max_dpd" type="number" placeholder="Max Days Past Due">
<input name="dpd_30" type="number" placeholder="30+ DPD Count">
<input name="dpd_60" type="number" placeholder="60+ DPD Count">
<input name="dpd_90" type="number" placeholder="90+ DPD Count">
<input name="months_since_default" type="number" placeholder="Months Since Default">
<input name="restructures" type="number" placeholder="Loan Restructures">
</div>
</div>

<div class="section">
<h3>💰 Financials</h3>
<div class="grid">
<input name="MonthlyIncome" type="number" placeholder="Monthly Income">
<input name="AnnualIncome" type="number" placeholder="Annual Income">
<input name="LoanAmount" type="number" placeholder="Requested Loan">
<input name="MonthlyLoanPayment" type="number" placeholder="Monthly Loan Payment">
<input name="MonthlyDebtPayments" type="number" placeholder="Other Monthly Debts">
</div>
</div>

<div class="section">
<h3>📋 Loan & Employment</h3>
<div class="grid">
<select name="EmploymentStatus"><option>Employed</option><option>Self-Employed</option><option>Unemployed</option></select>
<select name="EducationLevel"><option>Bachelor</option><option>Master</option><option>PhD</option></select>
<select name="LoanPurpose"><option>Home</option><option>Auto</option><option>Business</option></select>
</div>
</div>

<button type="submit">Assess Risk</button>
</form>
</div>
</body>
</html>
"""

# --------------------------------------------------
# Result Page
# --------------------------------------------------

RESULT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Risk Result</title>
<style>
body { font-family:Segoe UI; background:#f2f4f8; padding:20px; }
.card { max-width:900px; margin:auto; background:white; padding:30px; border-radius:14px; }
.header { background:{color}; color:white; padding:25px; border-radius:12px; text-align:center; }
.badge { font-size:1.4em; font-weight:bold; }
.section { margin-top:25px; }
</style>
</head>

<body>
<div class="card">
<div class="header">
<h2>{customer_id}</h2>
<p>Model Risk Score: {risk:.2%}</p>
<p>Behavioral Risk Index: {bri:.2%}</p>
<div class="badge">{decision}</div>
</div>

<div class="section">
<h3>Decision Rationale</h3>
<p>{reason}</p>
</div>

<div class="section">
<h3>Recommended Loan</h3>
<p style="font-size:2em;font-weight:bold;">₦{loan:,.2f}</p>
</div>

<p><i>Assessed on {time}</i></p>
<a href="/">Assess Another</a>
</div>
</body>
</html>
"""

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return HOME_TEMPLATE

@app.post("/assess", response_class=HTMLResponse)
def assess(
    customer_id: str = Form(...),
    Age: float = Form(...),
    Experience: float = Form(...),
    JobTenure: float = Form(...),
    MonthlyIncome: float = Form(...),
    AnnualIncome: float = Form(...),
    LoanAmount: float = Form(...),
    MonthlyLoanPayment: float = Form(...),
    MonthlyDebtPayments: float = Form(...),
    EmploymentStatus: str = Form(...),
    EducationLevel: str = Form(...),
    LoanPurpose: str = Form(...),
    max_dpd: int = Form(...),
    dpd_30: int = Form(...),
    dpd_60: int = Form(...),
    dpd_90: int = Form(...),
    months_since_default: int = Form(...),
    restructures: int = Form(...)
):

    features = {
        "Age": Age,
        "Experience": Experience,
        "JobTenure": JobTenure,
        "CreditScore": 650,
        "PaymentHistory": 0.9,
        "LengthOfCreditHistory": 5,
        "NumberOfOpenCreditLines": 3,
        "NumberOfCreditInquiries": 1,
        "LoanDuration": 36,
        "BaseInterestRate": 5.0,
        "InterestRate": 7.5,
        "TotalDebtToIncomeRatio": MonthlyDebtPayments / max(MonthlyIncome, 1),
        "MonthlyIncome_log": safe_log(MonthlyIncome),
        "AnnualIncome_log": safe_log(AnnualIncome),
        "MonthlyLoanPayment_log": safe_log(MonthlyLoanPayment),
        "LoanAmount_log": safe_log(LoanAmount),
        "MonthlyDebtPayments_log": safe_log(MonthlyDebtPayments),
        "EmploymentStatus": EmploymentStatus,
        "EducationLevel": EducationLevel,
        "MaritalStatus": "Single",
        "HomeOwnershipStatus": "Rent",
        "LoanPurpose": LoanPurpose
    }

    df = pd.DataFrame([features])[FEATURE_NAMES]
    model_risk = float(pipeline.predict(df)[0])

    bri = compute_behavioral_risk_index(
        max_dpd, dpd_30, dpd_60, dpd_90, months_since_default, restructures
    )

    if bri > 0.6:
        decision = "DECLINE – BEHAVIORAL RISK"
        reason = "Historical repayment behaviour indicates high default risk."
    elif bri > 0.4:
        decision = "MANUAL REVIEW"
        reason = "Mixed behavioural signals require analyst review."
    else:
        decision = "APPROVE"
        reason = "Consistent repayment behaviour observed."

    return RESULT_TEMPLATE.format(
        customer_id=customer_id,
        risk=model_risk,
        bri=bri,
        decision=decision,
        reason=reason,
        loan=LoanAmount * (1 - model_risk),
        color="#dc2626" if "DECLINE" in decision else "#f59e0b" if "REVIEW" in decision else "#16a34a",
        time=datetime.now().strftime("%Y-%m-%d %H:%M")
    )
