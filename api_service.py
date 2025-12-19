"""
FastAPI Web Application for Loan Risk Assessment
No database dependency – Render deployment ready
"""

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os
from fastapi.responses import HTMLResponse


app = FastAPI(title="Loan Risk Assessment API")

print("Loading model...")

with open("risk_assessment_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

print("✓ Model loaded successfully")

# --------------------------------------------------
# Config
# --------------------------------------------------

config = {
    "model_version": "1.1",
    "training_date": "2025-12-19",
    "feature_names": [
        "Age", "Experience", "JobTenure", "CreditScore", "PaymentHistory",
        "LengthOfCreditHistory", "NumberOfOpenCreditLines",
        "NumberOfCreditInquiries", "PreviousLoanDefaults",
        "BankruptcyHistory", "UtilityBillsPaymentHistory",
        "LoanDuration", "BaseInterestRate", "InterestRate",
        "TotalDebtToIncomeRatio", "MonthlyIncome_log",
        "AnnualIncome_log", "SavingsAccountBalance_log",
        "CheckingAccountBalance_log", "NetWorth_log",
        "TotalAssets_log", "TotalLiabilities_log",
        "MonthlyLoanPayment_log", "LoanAmount_log",
        "MonthlyDebtPayments_log", "EmploymentStatus",
        "EducationLevel", "MaritalStatus",
        "HomeOwnershipStatus", "LoanPurpose"
    ]
}

# --------------------------------------------------
# Business Logic
# --------------------------------------------------

def recommend_loan_amount(risk_score, annual_income, existing_monthly_debt, credit_score):
    base_multiplier = 0.3
    max_dti_ratio = 0.43

    risk_adjusted_multiplier = base_multiplier * (1 - risk_score)

    credit_adjustment = (
        1.2 if credit_score >= 750 else
        1.1 if credit_score >= 700 else
        1.0 if credit_score >= 650 else
        0.8 if credit_score >= 600 else
        0.6
    )

    monthly_income = annual_income / 12
    max_total_monthly_debt = monthly_income * max_dti_ratio
    available_payment = max_total_monthly_debt - existing_monthly_debt

    if available_payment <= 0:
        return {"recommended_max_loan": 0, "monthly_payment_capacity": 0}

    monthly_rate = 0.05 / 12
    n = 60

    max_from_payment = available_payment * (
        ((1 + monthly_rate) ** n - 1) /
        (monthly_rate * (1 + monthly_rate) ** n)
    )

    max_from_income = annual_income * risk_adjusted_multiplier * credit_adjustment
    recommended = min(max_from_payment, max_from_income, annual_income * 0.5)

    return {
        "recommended_max_loan": max(0, recommended),
        "monthly_payment_capacity": available_payment
    }

def get_risk_tier_info(risk_score):
    if risk_score < 0.3:
        return ("Low Risk", "A", 0.0, "AUTO_APPROVE", "Excellent credit profile", "#28a745")
    elif risk_score < 0.5:
        return ("Medium-Low Risk", "B", 1.0, "APPROVE", "Good credit profile", "#5cb85c")
    elif risk_score < 0.65:
        return ("Medium Risk", "C", 2.0, "MANUAL_REVIEW", "Acceptable risk", "#ffc107")
    elif risk_score < 0.8:
        return ("Medium-High Risk", "D", 3.5, "MANUAL_REVIEW_REQUIRED", "Elevated risk", "#ff9800")
    else:
        return ("High Risk", "E", 5.0, "DECLINE", "Significant risk", "#dc3545")



# --------------------------------------------------
# Routes
# --------------------------------------------------

# @app.get("/", response_class=HTMLResponse)
# def home():
#     return HOME_TEMPLATE  # noqa: F821

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
    PreviousLoanDefaults: float = Form(...),
    BankruptcyHistory: float = Form(...),
    UtilityBillsPaymentHistory: float = Form(...),
    LoanDuration: float = Form(...),
    BaseInterestRate: float = Form(...),
    InterestRate: float = Form(...),
    TotalDebtToIncomeRatio: float = Form(...),
    MonthlyIncome_log: float = Form(...),
    AnnualIncome_log: float = Form(...),
    SavingsAccountBalance_log: float = Form(...),
    CheckingAccountBalance_log: float = Form(...),
    NetWorth_log: float = Form(...),
    TotalAssets_log: float = Form(...),
    TotalLiabilities_log: float = Form(...),
    MonthlyLoanPayment_log: float = Form(...),
    LoanAmount_log: float = Form(...),
    MonthlyDebtPayments_log: float = Form(...),
    EmploymentStatus: str = Form(...),
    EducationLevel: str = Form(...),
    MaritalStatus: str = Form(...),
    HomeOwnershipStatus: str = Form(...),
    LoanPurpose: str = Form(...)
):
    features = locals()
    features.pop("customer_id")
    features.pop("request", None)

    df = pd.DataFrame([features])[config["feature_names"]]
    risk_score = pipeline.predict(df)[0]

    tier, code, rate_adj, decision, desc, color = get_risk_tier_info(risk_score)

    annual_income = np.exp(AnnualIncome_log)
    monthly_debt = np.exp(MonthlyDebtPayments_log)

    loan = recommend_loan_amount(risk_score, annual_income, monthly_debt, CreditScore)

    return RESULT_TEMPLATE.format(
        customer_id=customer_id,
        risk_score=risk_score,
        risk_tier=tier,
        risk_code=code,
        risk_description=desc,
        risk_color=color,
        decision=decision,
        decision_color=color,
        decision_message=decision.replace("_", " "),
        max_loan=loan["recommended_max_loan"],
        monthly_capacity=loan["monthly_payment_capacity"],
        estimated_payment=loan["monthly_payment_capacity"] * 0.9,
        base_rate=5.0,
        adjusted_rate=5.0 + rate_adj,
        annual_income=annual_income,
        credit_score=int(CreditScore),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.get("/health")
def health():
    return JSONResponse({
        "status": "healthy",
        "model_version": config["model_version"],
        "timestamp": datetime.now().isoformat()
    })

# --------------------------------------------------
# Render entry point
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)