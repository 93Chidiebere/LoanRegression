"""
FastAPI Web Application for Loan Risk Assessment
• User enters NAIRA values
• Model internally uses LOG values
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
    """Convert naira amount to log safely."""
    return float(np.log(max(x, 1.0)))

def safe_exp(x: float) -> float:
    """Convert log back to naira."""
    return float(np.exp(x))

# --------------------------------------------------
# App & Model
# --------------------------------------------------

app = FastAPI(title="Loan Risk Assessment System")

with open("risk_assessment_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

FEATURE_NAMES = [
    "Age", "Experience", "JobTenure", "CreditScore", "PaymentHistory",
    "LengthOfCreditHistory", "NumberOfOpenCreditLines",
    "NumberOfCreditInquiries", "PreviousLoanDefaults",
    "BankruptcyHistory", "UtilityBillsPaymentHistory",
    "LoanDuration", "BaseInterestRate", "InterestRate",
    "TotalDebtToIncomeRatio",
    "MonthlyIncome_log", "AnnualIncome_log",
    "SavingsAccountBalance_log", "CheckingAccountBalance_log",
    "NetWorth_log", "TotalAssets_log", "TotalLiabilities_log",
    "MonthlyLoanPayment_log", "LoanAmount_log",
    "MonthlyDebtPayments_log",
    "EmploymentStatus", "EducationLevel", "MaritalStatus",
    "HomeOwnershipStatus", "LoanPurpose"
]

# --------------------------------------------------
# UI – HOME PAGE
# --------------------------------------------------

HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Loan Risk Assessment</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body {
    font-family: Segoe UI, Arial, sans-serif;
    background: #f2f4f8;
    padding: 20px;
}
.container {
    max-width: 1000px;
    margin: auto;
    background: white;
    padding: 35px;
    border-radius: 14px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.12);
}
h1 {
    text-align: center;
    margin-bottom: 10px;
}
.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 30px;
}
.section {
    margin-bottom: 30px;
}
.section h3 {
    border-bottom: 2px solid #4f46e5;
    padding-bottom: 8px;
    color: #4f46e5;
}
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 15px;
}
label {
    font-weight: 600;
    font-size: 0.9em;
}
input, select {
    padding: 10px;
    border-radius: 6px;
    border: 1px solid #ccc;
}
button {
    width: 100%;
    padding: 15px;
    font-size: 1.1em;
    font-weight: bold;
    background: #4f46e5;
    color: white;
    border: none;
    border-radius: 50px;
    cursor: pointer;
}
button:hover {
    background: #4338ca;
}
.note {
    background: #eef2ff;
    padding: 12px;
    border-left: 4px solid #4f46e5;
    margin-bottom: 25px;
}
</style>
</head>

<body>
<div class="container">
<h1>Loan Risk Assessment</h1>
<p class="subtitle">Credit Risk, Pricing & Decision Support</p>

<div class="note">
<b>Note:</b> All financial values should be entered in <b>Naira (₦)</b>.
</div>

<form method="post" action="/assess">

<div class="section">
<h3>👤 Personal Information</h3>
<div class="grid">
<input type="text" name="customer_id" placeholder="Customer ID" required>
<input type="number" name="Age" placeholder="Age" required>
<input type="number" name="Experience" placeholder="Years of Experience" required>
</div>
</div>

<div class="section">
<h3>💼 Employment & Education</h3>
<div class="grid">
<select name="EmploymentStatus">
<option>Employed</option><option>Self-Employed</option><option>Unemployed</option>
</select>
<select name="EducationLevel">
<option>Bachelor</option><option>Master</option><option>PhD</option>
</select>
<input type="number" name="JobTenure" placeholder="Job Tenure (years)">
</div>
</div>

<div class="section">
<h3>💳 Credit Profile</h3>
<div class="grid">
<input type="number" name="CreditScore" placeholder="Credit Score">
<input type="number" step="0.01" name="PaymentHistory" placeholder="Payment History (0–1)">
<input type="number" name="LengthOfCreditHistory" placeholder="Credit History (years)">
<input type="number" name="NumberOfOpenCreditLines" placeholder="Open Credit Lines">
<input type="number" name="NumberOfCreditInquiries" placeholder="Recent Inquiries">
</div>
</div>

<div class="section">
<h3>🏠 Loan Information</h3>
<div class="grid">
<input type="number" name="LoanDuration" placeholder="Loan Duration (months)">
<input type="number" step="0.1" name="BaseInterestRate" placeholder="Base Rate (%)">
<input type="number" step="0.1" name="InterestRate" placeholder="Applied Rate (%)">
<select name="LoanPurpose">
<option>Home</option><option>Auto</option><option>Business</option>
</select>
</div>
</div>

<div class="section">
<h3>💰 Financial Information (₦)</h3>
<div class="grid">
<input type="number" name="MonthlyIncome" placeholder="Monthly Income">
<input type="number" name="AnnualIncome" placeholder="Annual Income">
<input type="number" name="LoanAmount" placeholder="Requested Loan Amount">
<input type="number" name="MonthlyLoanPayment" placeholder="Monthly Loan Payment">
<input type="number" name="MonthlyDebtPayments" placeholder="Other Monthly Debts">
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
body { font-family: Segoe UI; background: #f2f4f8; padding: 20px; }
.card {
    max-width: 800px; margin: auto; background: white;
    padding: 30px; border-radius: 14px;
}
.header {
    background: {color}; color: white;
    padding: 25px; border-radius: 12px; text-align: center;
}
.amount { font-size: 2em; font-weight: bold; }
</style>
</head>

<body>
<div class="card">
<div class="header">
<h2>{customer_id}</h2>
<p>Risk Score: {risk:.2%}</p>
<p>{tier} (Grade {grade})</p>
</div>

<h3>Decision</h3>
<p><b>{decision}</b> – {description}</p>

<h3>Recommended Loan</h3>
<p class="amount">₦{loan:,.2f}</p>

<h3>Interest Rate</h3>
<p>{rate}%</p>

<p><i>Assessed on {time}</i></p>

<a href="/">Assess Another Customer</a>
</div>
</body>
</html>
"""

# --------------------------------------------------
# Policy Logic
# --------------------------------------------------

def risk_policy(score):
    if score < 0.3:
        return ("Low Risk", "A", "APPROVE", "Excellent profile", "#16a34a", 5.0)
    elif score < 0.6:
        return ("Medium Risk", "B", "REVIEW", "Moderate risk", "#f59e0b", 7.0)
    else:
        return ("High Risk", "C", "DECLINE", "High default risk", "#dc2626", 10.0)

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
    CreditScore: float = Form(...),
    PaymentHistory: float = Form(...),
    LengthOfCreditHistory: float = Form(...),
    NumberOfOpenCreditLines: float = Form(...),
    NumberOfCreditInquiries: float = Form(...),
    LoanDuration: float = Form(...),
    BaseInterestRate: float = Form(...),
    InterestRate: float = Form(...),
    MonthlyIncome: float = Form(...),
    AnnualIncome: float = Form(...),
    LoanAmount: float = Form(...),
    MonthlyLoanPayment: float = Form(...),
    MonthlyDebtPayments: float = Form(...),
    EmploymentStatus: str = Form(...),
    EducationLevel: str = Form(...),
    LoanPurpose: str = Form(...)
):
    features = {
        "Age": Age,
        "Experience": Experience,
        "JobTenure": JobTenure,
        "CreditScore": CreditScore,
        "PaymentHistory": PaymentHistory,
        "LengthOfCreditHistory": LengthOfCreditHistory,
        "NumberOfOpenCreditLines": NumberOfOpenCreditLines,
        "NumberOfCreditInquiries": NumberOfCreditInquiries,
        "LoanDuration": LoanDuration,
        "BaseInterestRate": BaseInterestRate,
        "InterestRate": InterestRate,
        "TotalDebtToIncomeRatio": MonthlyDebtPayments / max(MonthlyIncome, 1),
        "MonthlyIncome_log": safe_log(MonthlyIncome),
        "AnnualIncome_log": safe_log(AnnualIncome),
        "SavingsAccountBalance_log": safe_log(1),
        "CheckingAccountBalance_log": safe_log(1),
        "NetWorth_log": safe_log(1),
        "TotalAssets_log": safe_log(1),
        "TotalLiabilities_log": safe_log(1),
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
    risk = float(pipeline.predict(df)[0])

    tier, grade, decision, desc, color, rate = risk_policy(risk)

    return RESULT_TEMPLATE.format(
        customer_id=customer_id,
        risk=risk,
        tier=tier,
        grade=grade,
        decision=decision,
        description=desc,
        loan=LoanAmount * (1 - risk),
        rate=rate,
        color=color,
        time=datetime.now().strftime("%Y-%m-%d %H:%M")
    )
