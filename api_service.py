# """
# Simple Flask API for Loan Risk Assessment
# """

# from flask import Flask, request, jsonify
# import pandas as pd
# import numpy as np
# import pickle
# import psycopg2
# from datetime import datetime
# import json

# print("Loading model...")

# # Load model - it's right in the main folder!
# with open('risk_assessment_pipeline.pkl', 'rb') as f:
#     pipeline = pickle.load(f)

# print("✓ Model loaded successfully")

# # Create config (since you don't have deployment_config.pkl, we'll create it here)
# config = {
#     'model_version': '1.0',
#     'training_date': '2025-12-18',
#     'c_over': 1.0,
#     'c_under': 5.0,
#     'feature_names': [
#         'Age', 'Experience', 'JobTenure', 'CreditScore', 'PaymentHistory',
#         'LengthOfCreditHistory', 'NumberOfOpenCreditLines', 'NumberOfCreditInquiries',
#         'PreviousLoanDefaults', 'BankruptcyHistory', 'UtilityBillsPaymentHistory',
#         'LoanDuration', 'BaseInterestRate', 'InterestRate', 'TotalDebtToIncomeRatio',
#         'MonthlyIncome_log', 'AnnualIncome_log', 'SavingsAccountBalance_log',
#         'CheckingAccountBalance_log', 'NetWorth_log', 'TotalAssets_log',
#         'TotalLiabilities_log', 'MonthlyLoanPayment_log', 'LoanAmount_log',
#         'MonthlyDebtPayments_log', 'EmploymentStatus', 'EducationLevel',
#         'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose'
#     ]
# }

# # Database config
# DB_CONFIG = {
#     "host": "localhost",
#     "database": "ml_monitoring",
#     "user": "ml_user",
#     "password": "StrongPassword123",
#     "port": 5434
# }

# app = Flask(__name__)


# def get_db_connection():
#     try:
#         return psycopg2.connect(**DB_CONFIG)
#     except Exception as e:
#         print(f"DB Error: {e}")
#         return None

# def recommend_loan_amount(risk_score, annual_income, existing_monthly_debt, credit_score):
#     base_multiplier = 0.3
#     max_dti_ratio = 0.43
    
#     risk_adjusted_multiplier = base_multiplier * (1 - risk_score)
    
#     if credit_score >= 750:
#         credit_adjustment = 1.2
#     elif credit_score >= 700:
#         credit_adjustment = 1.1
#     elif credit_score >= 650:
#         credit_adjustment = 1.0
#     elif credit_score >= 600:
#         credit_adjustment = 0.8
#     else:
#         credit_adjustment = 0.6
    
#     monthly_income = annual_income / 12
#     max_total_monthly_debt = monthly_income * max_dti_ratio
#     available_monthly_payment = max_total_monthly_debt - existing_monthly_debt
    
#     if available_monthly_payment <= 0:
#         return {
#             'recommended_max_loan': 0,
#             'monthly_payment_capacity': 0,
#             'reason': 'DTI at maximum'
#         }
    
#     monthly_rate = 0.05 / 12
#     n_payments = 60
    
#     max_loan_from_payment = available_monthly_payment * (
#         ((1 + monthly_rate) ** n_payments - 1) / 
#         (monthly_rate * (1 + monthly_rate) ** n_payments)
#     )
    
#     max_loan_from_income = annual_income * risk_adjusted_multiplier * credit_adjustment
#     recommended_loan = min(max_loan_from_payment, max_loan_from_income)
#     absolute_max = annual_income * 0.5
#     recommended_loan = min(recommended_loan, absolute_max)
    
#     return {
#         'recommended_max_loan': max(0, recommended_loan),
#         'monthly_payment_capacity': available_monthly_payment
#     }

# def get_risk_tier_info(risk_score):
#     if risk_score < 0.3:
#         return {
#             'tier': 'Low Risk',
#             'tier_code': 'A',
#             'interest_rate_adjustment': 0.0,
#             'approval_recommendation': 'AUTO_APPROVE',
#             'description': 'Excellent credit profile'
#         }
#     elif risk_score < 0.5:
#         return {
#             'tier': 'Medium-Low Risk',
#             'tier_code': 'B',
#             'interest_rate_adjustment': 1.0,
#             'approval_recommendation': 'APPROVE',
#             'description': 'Good credit profile'
#         }
#     elif risk_score < 0.65:
#         return {
#             'tier': 'Medium Risk',
#             'tier_code': 'C',
#             'interest_rate_adjustment': 2.0,
#             'approval_recommendation': 'MANUAL_REVIEW',
#             'description': 'Acceptable risk'
#         }
#     elif risk_score < 0.8:
#         return {
#             'tier': 'Medium-High Risk',
#             'tier_code': 'D',
#             'interest_rate_adjustment': 3.5,
#             'approval_recommendation': 'MANUAL_REVIEW_REQUIRED',
#             'description': 'Elevated risk'
#         }
#     else:
#         return {
#             'tier': 'High Risk',
#             'tier_code': 'E',
#             'interest_rate_adjustment': 5.0,
#             'approval_recommendation': 'DECLINE',
#             'description': 'Significant risk'
#         }

# def log_prediction_to_db(customer_id, features, risk_score, recommended_loan, approval_decision, timestamp):
#     try:
#         conn = get_db_connection()
#         if not conn:
#             return False
        
#         cursor = conn.cursor()
#         cursor.execute("""
#             INSERT INTO predictions 
#             (customer_id, timestamp, input_features, risk_score, recommended_loan, approval_decision)
#             VALUES (%s, %s, %s, %s, %s, %s)
#         """, (customer_id, timestamp, json.dumps(features), float(risk_score), 
#               float(recommended_loan), approval_decision))
        
#         conn.commit()
#         cursor.close()
#         conn.close()
#         return True
#     except Exception as e:
#         print(f"Log error: {e}")
#         return False

# @app.route('/')
# def home():
#     return jsonify({
#         'service': 'Loan Risk Assessment API',
#         'version': config['model_version'],
#         'endpoints': {
#             'predict': '/predict (POST)',
#             'health': '/health (GET)'
#         }
#     })

# @app.route('/health')
# def health():
#     conn = get_db_connection()
#     db_status = "connected" if conn else "disconnected"
#     if conn:
#         conn.close()
    
#     return jsonify({
#         'status': 'healthy',
#         'model_version': config['model_version'],
#         'database': db_status,
#         'timestamp': datetime.now().isoformat()
#     })

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         customer_id = data.get('customer_id', 'UNKNOWN')
#         features = data['features']
        
#         # Convert to DataFrame
#         customer_df = pd.DataFrame([features])
#         customer_df = customer_df[config['feature_names']]
        
#         # Predict
#         risk_score = pipeline.predict(customer_df)[0] # type: ignore
        
#         # Get risk tier
#         risk_info = get_risk_tier_info(risk_score)
        
#         # Get financials
#         annual_income = np.exp(features['AnnualIncome_log'])
#         monthly_debt = np.exp(features['MonthlyDebtPayments_log'])
#         credit_score = features['CreditScore']
        
#         # Get loan recommendation
#         loan_rec = recommend_loan_amount(risk_score, annual_income, monthly_debt, credit_score)
        
#         timestamp = datetime.now()
        
#         response = {
#             'customer_id': customer_id,
#             'risk_assessment': {
#                 'risk_score': float(risk_score),
#                 'risk_tier': risk_info['tier'],
#                 'risk_tier_code': risk_info['tier_code'],
#                 'description': risk_info['description']
#             },
#             'loan_recommendation': {
#                 'max_approved_amount': float(loan_rec['recommended_max_loan']),
#                 'monthly_payment_capacity': float(loan_rec['monthly_payment_capacity']),
#                 'estimated_monthly_payment': float(loan_rec['monthly_payment_capacity'] * 0.9)
#             },
#             'lending_terms': {
#                 'base_interest_rate': 5.0,
#                 'risk_adjusted_rate': 5.0 + risk_info['interest_rate_adjustment'],
#                 'approval_decision': risk_info['approval_recommendation']
#             },
#             'metadata': {
#                 'model_version': config['model_version'],
#                 'prediction_timestamp': timestamp.isoformat(),
#                 'annual_income': float(annual_income),
#                 'credit_score': int(credit_score)
#             }
#         }
        
#         # Log to database
#         log_prediction_to_db(
#             customer_id, features, risk_score,
#             loan_rec['recommended_max_loan'],
#             risk_info['approval_recommendation'],
#             timestamp
#         )
        
#         return jsonify(response)
        
#     except Exception as e:
#         return jsonify({
#             'error': str(e),
#             'message': 'Prediction failed'
#         }), 400

# if __name__ == '__main__':
#     print("="*70)
#     print("LOAN RISK ASSESSMENT API")
#     print("="*70)
#     print(f"Model version: {config['model_version']}")
#     print("="*70)
#     print("\nStarting Flask server...")
#     print("API will be available at: http://127.0.0.1:5000")
#     print("="*70)
    
#     app.run(host='0.0.0.0', port=5000, debug=True)

"""
Flask Web Application for Loan Risk Assessment
With HTML interface for manual input
"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import pickle
import psycopg2
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Loading model...")

# Load model
with open('risk_assessment_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

print("✓ Model loaded successfully")

# Config
config = {
    'model_version': '1.0',
    'training_date': '2025-12-18',
    'c_over': 1.0,
    'c_under': 5.0,
    'feature_names': [
        'Age', 'Experience', 'JobTenure', 'CreditScore', 'PaymentHistory',
        'LengthOfCreditHistory', 'NumberOfOpenCreditLines', 'NumberOfCreditInquiries',
        'PreviousLoanDefaults', 'BankruptcyHistory', 'UtilityBillsPaymentHistory',
        'LoanDuration', 'BaseInterestRate', 'InterestRate', 'TotalDebtToIncomeRatio',
        'MonthlyIncome_log', 'AnnualIncome_log', 'SavingsAccountBalance_log',
        'CheckingAccountBalance_log', 'NetWorth_log', 'TotalAssets_log',
        'TotalLiabilities_log', 'MonthlyLoanPayment_log', 'LoanAmount_log',
        'MonthlyDebtPayments_log', 'EmploymentStatus', 'EducationLevel',
        'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose'
    ]
}

# Database config
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_DATABASE", "ml_monitoring"),
    "user": os.getenv("DB_USER", "ml_user"),
    "password": os.getenv("DB_PASSWORD", ""),
    "port": int(os.getenv("DB_PORT", "5434"))
}

app = Flask(__name__)

def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"DB Error: {e}")
        return None

def recommend_loan_amount(risk_score, annual_income, existing_monthly_debt, credit_score):
    base_multiplier = 0.3
    max_dti_ratio = 0.43
    
    risk_adjusted_multiplier = base_multiplier * (1 - risk_score)
    
    if credit_score >= 750:
        credit_adjustment = 1.2
    elif credit_score >= 700:
        credit_adjustment = 1.1
    elif credit_score >= 650:
        credit_adjustment = 1.0
    elif credit_score >= 600:
        credit_adjustment = 0.8
    else:
        credit_adjustment = 0.6
    
    monthly_income = annual_income / 12
    max_total_monthly_debt = monthly_income * max_dti_ratio
    available_monthly_payment = max_total_monthly_debt - existing_monthly_debt
    
    if available_monthly_payment <= 0:
        return {
            'recommended_max_loan': 0,
            'monthly_payment_capacity': 0,
            'reason': 'DTI at maximum'
        }
    
    monthly_rate = 0.05 / 12
    n_payments = 60
    
    max_loan_from_payment = available_monthly_payment * (
        ((1 + monthly_rate) ** n_payments - 1) / 
        (monthly_rate * (1 + monthly_rate) ** n_payments)
    )
    
    max_loan_from_income = annual_income * risk_adjusted_multiplier * credit_adjustment
    recommended_loan = min(max_loan_from_payment, max_loan_from_income)
    absolute_max = annual_income * 0.5
    recommended_loan = min(recommended_loan, absolute_max)
    
    return {
        'recommended_max_loan': max(0, recommended_loan),
        'monthly_payment_capacity': available_monthly_payment
    }

def get_risk_tier_info(risk_score):
    if risk_score < 0.3:
        return {
            'tier': 'Low Risk',
            'tier_code': 'A',
            'interest_rate_adjustment': 0.0,
            'approval_recommendation': 'AUTO_APPROVE',
            'description': 'Excellent credit profile',
            'color': '#28a745'
        }
    elif risk_score < 0.5:
        return {
            'tier': 'Medium-Low Risk',
            'tier_code': 'B',
            'interest_rate_adjustment': 1.0,
            'approval_recommendation': 'APPROVE',
            'description': 'Good credit profile',
            'color': '#5cb85c'
        }
    elif risk_score < 0.65:
        return {
            'tier': 'Medium Risk',
            'tier_code': 'C',
            'interest_rate_adjustment': 2.0,
            'approval_recommendation': 'MANUAL_REVIEW',
            'description': 'Acceptable risk',
            'color': '#ffc107'
        }
    elif risk_score < 0.8:
        return {
            'tier': 'Medium-High Risk',
            'tier_code': 'D',
            'interest_rate_adjustment': 3.5,
            'approval_recommendation': 'MANUAL_REVIEW_REQUIRED',
            'description': 'Elevated risk',
            'color': '#ff9800'
        }
    else:
        return {
            'tier': 'High Risk',
            'tier_code': 'E',
            'interest_rate_adjustment': 5.0,
            'approval_recommendation': 'DECLINE',
            'description': 'Significant risk',
            'color': '#dc3545'
        }

def log_prediction_to_db(customer_id, features, risk_score, recommended_loan, approval_decision, timestamp):
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions 
            (customer_id, timestamp, input_features, risk_score, recommended_loan, approval_decision)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (customer_id, timestamp, json.dumps(features), float(risk_score), 
              float(recommended_loan), approval_decision))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Log error: {e}")
        return False

# HTML Templates
HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Loan Risk Assessment System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .form-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .form-section h3 {
            color: #667eea;
            margin-bottom: 15px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
        }
        
        label {
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
            font-size: 0.9em;
        }
        
        input, select {
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn-submit {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 50px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn-submit:active {
            transform: translateY(0);
        }
        
        .info-box {
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .info-box p {
            margin: 5px 0;
            color: #333;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #667eea;
            font-size: 1.2em;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏦 Loan Risk Assessment</h1>
        <p class="subtitle">AI-Powered Credit Risk Analysis System</p>
        
        <div class="info-box">
            <p><strong>ℹ️ Instructions:</strong></p>
            <p>• Fill in all customer information below</p>
            <p>• Use log values for financial amounts (e.g., log(₦50000) ≈ 10.82)</p>
            <p>• Click "Assess Risk" to get instant decision</p>
        </div>
        
        <form id="assessmentForm" method="POST" action="/assess">
            <!-- Personal Information -->
            <div class="form-section">
                <h3>👤 Personal Information</h3>
                <div class="form-row">
                    <div class="form-group">
                        <label>Customer ID:</label>
                        <input type="text" name="customer_id" value="CUST001" required>
                    </div>
                    <div class="form-group">
                        <label>Age:</label>
                        <input type="number" name="Age" value="35" min="18" max="100" required>
                    </div>
                    <div class="form-group">
                        <label>Experience (years):</label>
                        <input type="number" name="Experience" value="10" min="0" max="50" required>
                    </div>
                </div>
            </div>
            
            <!-- Employment -->
            <div class="form-section">
                <h3>💼 Employment Details</h3>
                <div class="form-row">
                    <div class="form-group">
                        <label>Employment Status:</label>
                        <select name="EmploymentStatus" required>
                            <option value="Employed">Employed</option>
                            <option value="Self-Employed">Self-Employed</option>
                            <option value="Unemployed">Unemployed</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Job Tenure (years):</label>
                        <input type="number" name="JobTenure" value="3" min="0" max="40" required>
                    </div>
                    <div class="form-group">
                        <label>Education Level:</label>
                        <select name="EducationLevel" required>
                            <option value="High School">High School</option>
                            <option value="Associate">Associate</option>
                            <option value="Bachelor">Bachelor</option>
                            <option value="Master">Master</option>
                            <option value="PhD">PhD</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <!-- Credit Information -->
            <div class="form-section">
                <h3>💳 Credit Information</h3>
                <div class="form-row">
                    <div class="form-group">
                        <label>Credit Score:</label>
                        <input type="number" name="CreditScore" value="680" min="300" max="850" required>
                    </div>
                    <div class="form-group">
                        <label>Payment History (0-1):</label>
                        <input type="number" name="PaymentHistory" value="0.95" step="0.01" min="0" max="1" required>
                    </div>
                    <div class="form-group">
                        <label>Credit History Length (years):</label>
                        <input type="number" name="LengthOfCreditHistory" value="8" min="0" max="50" required>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Open Credit Lines:</label>
                        <input type="number" name="NumberOfOpenCreditLines" value="3" min="0" max="50" required>
                    </div>
                    <div class="form-group">
                        <label>Credit Inquiries:</label>
                        <input type="number" name="NumberOfCreditInquiries" value="1" min="0" max="20" required>
                    </div>
                    <div class="form-group">
                        <label>Previous Loan Defaults:</label>
                        <input type="number" name="PreviousLoanDefaults" value="0" min="0" max="10" required>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Bankruptcy History:</label>
                        <input type="number" name="BankruptcyHistory" value="0" min="0" max="5" required>
                    </div>
                    <div class="form-group">
                        <label>Utility Bills Payment (0-1):</label>
                        <input type="number" name="UtilityBillsPaymentHistory" value="1.0" step="0.01" min="0" max="1" required>
                    </div>
                </div>
            </div>
            
            <!-- Loan Details -->
            <div class="form-section">
                <h3>🏠 Loan Details</h3>
                <div class="form-row">
                    <div class="form-group">
                        <label>Loan Duration (months):</label>
                        <input type="number" name="LoanDuration" value="36" min="6" max="360" required>
                    </div>
                    <div class="form-group">
                        <label>Base Interest Rate (%):</label>
                        <input type="number" name="BaseInterestRate" value="4.5" step="0.1" min="0" max="30" required>
                    </div>
                    <div class="form-group">
                        <label>Interest Rate (%):</label>
                        <input type="number" name="InterestRate" value="5.2" step="0.1" min="0" max="30" required>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Loan Purpose:</label>
                        <select name="LoanPurpose" required>
                            <option value="Home">Home</option>
                            <option value="Auto">Auto</option>
                            <option value="Education">Education</option>
                            <option value="Business">Business</option>
                            <option value="Other">Other</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Debt-to-Income Ratio:</label>
                        <input type="number" name="TotalDebtToIncomeRatio" value="0.35" step="0.01" min="0" max="2" required>
                    </div>
                </div>
            </div>
            
            <!-- Financial Information (Log Values) -->
            <div class="form-section">
                <h3>💰 Financial Information (Log Values)</h3>
                <div class="form-row">
                    <div class="form-group">
                        <label>Monthly Income (log):</label>
                        <input type="number" name="MonthlyIncome_log" value="10.5" step="0.01" required>
                    </div>
                    <div class="form-group">
                        <label>Annual Income (log):</label>
                        <input type="number" name="AnnualIncome_log" value="11.0" step="0.01" required>
                    </div>
                    <div class="form-group">
                        <label>Savings Balance (log):</label>
                        <input type="number" name="SavingsAccountBalance_log" value="9.5" step="0.01" required>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Checking Balance (log):</label>
                        <input type="number" name="CheckingAccountBalance_log" value="8.2" step="0.01" required>
                    </div>
                    <div class="form-group">
                        <label>Net Worth (log):</label>
                        <input type="number" name="NetWorth_log" value="11.5" step="0.01" required>
                    </div>
                    <div class="form-group">
                        <label>Total Assets (log):</label>
                        <input type="number" name="TotalAssets_log" value="12.0" step="0.01" required>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Total Liabilities (log):</label>
                        <input type="number" name="TotalLiabilities_log" value="10.8" step="0.01" required>
                    </div>
                    <div class="form-group">
                        <label>Monthly Loan Payment (log):</label>
                        <input type="number" name="MonthlyLoanPayment_log" value="7.5" step="0.01" required>
                    </div>
                    <div class="form-group">
                        <label>Loan Amount (log):</label>
                        <input type="number" name="LoanAmount_log" value="10.0" step="0.01" required>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Monthly Debt Payments (log):</label>
                        <input type="number" name="MonthlyDebtPayments_log" value="7.2" step="0.01" required>
                    </div>
                </div>
            </div>
            
            <!-- Personal Details -->
            <div class="form-section">
                <h3>👨‍👩‍👧‍👦 Personal Details</h3>
                <div class="form-row">
                    <div class="form-group">
                        <label>Marital Status:</label>
                        <select name="MaritalStatus" required>
                            <option value="Single">Single</option>
                            <option value="Married">Married</option>
                            <option value="Divorced">Divorced</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Home Ownership:</label>
                        <select name="HomeOwnershipStatus" required>
                            <option value="Rent">Rent</option>
                            <option value="Own">Own</option>
                            <option value="Mortgage">Mortgage</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <button type="submit" class="btn-submit">🔍 Assess Risk & Get Decision</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing customer data...</p>
        </div>
    </div>
    
    <script>
        document.getElementById('assessmentForm').onsubmit = function() {
            document.getElementById('loading').style.display = 'block';
        };
    </script>
</body>
</html>
"""

RESULT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Risk Assessment Result</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        
        .result-header {
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, {{ risk_color }} 0%, {{ risk_color }}dd 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        
        .result-header h2 {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .risk-score {
            font-size: 3em;
            font-weight: bold;
            margin: 20px 0;
        }
        
        .tier-badge {
            display: inline-block;
            padding: 10px 30px;
            background: rgba(255,255,255,0.3);
            border-radius: 50px;
            font-size: 1.2em;
            font-weight: 600;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .info-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid {{ risk_color }};
        }
        
        .info-card h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        
        .info-item {
            margin: 10px 0;
            color: #555;
        }
        
        .info-item strong {
            color: #333;
        }
        
        .amount {
            font-size: 1.8em;
            color: {{ risk_color }};
            font-weight: bold;
            margin: 10px 0;
        }
        
        .decision-box {
            background: {{ decision_color }};
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            margin: 30px 0;
        }
        
        .decision-box h2 {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .btn-group {
            display: flex;
            gap: 15px;
            margin-top: 30px;
        }
        
        .btn {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 50px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Risk Assessment Result</h1>
        
        <div class="result-header">
            <h2>{{ customer_id }}</h2>
            <div class="risk-score">{{ "%.2f"|format(risk_score * 100) }}%</div>
            <div class="tier-badge">{{ risk_tier }} ({{ risk_code }})</div>
            <p style="margin-top: 15px; font-size: 1.1em;">{{ risk_description }}</p>
        </div>
        
        <div class="decision-box">
            <h2>{{ decision }}</h2>
            <p style="font-size: 1.2em; margin-top: 10px;">{{ decision_message }}</p>
        </div>
        
        <div class="info-grid">
            <div class="info-card">
                <h3>💰 Loan Recommendation</h3>
                <div class="amount">₦{{ "{:,.2f}".format(max_loan) }}</div>
                <div class="info-item">
                    <strong>Monthly Payment Capacity:</strong><br>
                    ₦{{ "{:,.2f}".format(monthly_capacity) }}
                </div>
                <div class="info-item">
                    <strong>Estimated Monthly Payment:</strong><br>
                    ₦{{ "{:,.2f}".format(estimated_payment) }}
                </div>
            </div>
            
            <div class="info-card">
                <h3>💳 Interest Rates</h3>
                <div class="info-item">
                    <strong>Base Rate:</strong> {{ base_rate }}%
                </div>
                <div class="info-item">
                    <strong>Risk-Adjusted Rate:</strong>
                    <div class="amount" style="font-size: 1.4em;">{{ adjusted_rate }}%</div>
                </div>
                <div class="info-item">
                    <strong>Rate Increase:</strong> +{{ "%.1f"|format(adjusted_rate - base_rate) }}%
                </div>
            </div>
            
            <div class="info-card">
                <h3>📋 Customer Profile</h3>
                <div class="info-item">
                    <strong>Annual Income:</strong><br>
                    ₦{{ "{:,.2f}".format(annual_income) }}
                </div>
                <div class="info-item">
                    <strong>Credit Score:</strong> {{ credit_score }}
                </div>
                <div class="info-item">
                    <strong>Assessment Time:</strong><br>
                    {{ timestamp }}
                </div>
            </div>
        </div>
        
        <div class="btn-group">
            <button class="btn btn-primary" onclick="window.location.href='/'">
                🔄 Assess Another Customer
            </button>
            <button class="btn btn-secondary" onclick="window.print()">
                🖨️ Print Report
            </button>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    """Home page with input form"""
    return render_template_string(HOME_TEMPLATE)

@app.route('/assess', methods=['POST'])
def assess():
    """Process form submission and show results"""
    try:
        # Get form data
        customer_id = request.form.get('customer_id')
        
        # Build features dictionary
        features = {}
        for field in config['feature_names']:
            value = request.form.get(field)
            
            # Convert to appropriate type
            if field in ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']:
                features[field] = value
            else:
                features[field] = float(value)
        
        # Convert to DataFrame
        customer_df = pd.DataFrame([features])
        customer_df = customer_df[config['feature_names']]
        
        # Predict risk score
        risk_score = pipeline.predict(customer_df)[0]
        
        # Get risk tier info
        risk_info = get_risk_tier_info(risk_score)
        
        # Get financial data
        annual_income = np.exp(features['AnnualIncome_log'])
        monthly_debt = np.exp(features['MonthlyDebtPayments_log'])
        credit_score = features['CreditScore']
        
        # Get loan recommendation
        loan_rec = recommend_loan_amount(risk_score, annual_income, monthly_debt, credit_score)
        
        # Determine decision color
        if risk_info['approval_recommendation'] in ['AUTO_APPROVE', 'APPROVE']:
            decision_color = '#28a745'
            decision_message = 'Loan application can be approved'
        elif 'MANUAL_REVIEW' in risk_info['approval_recommendation']:
            decision_color = '#ffc107'
            decision_message = 'Requires manual review by loan officer'
        else:
            decision_color = '#dc3545'
            decision_message = 'Loan application should be declined'
        
        timestamp = datetime.now()
        
        # Log to database
        log_prediction_to_db(
            customer_id, features, risk_score,
            loan_rec['recommended_max_loan'],
            risk_info['approval_recommendation'],
            timestamp
        )
        
        # Render result page
        return render_template_string(
            RESULT_TEMPLATE,
            customer_id=customer_id,
            risk_score=risk_score,
            risk_tier=risk_info['tier'],
            risk_code=risk_info['tier_code'],
            risk_description=risk_info['description'],
            risk_color=risk_info['color'],
            decision=risk_info['approval_recommendation'],
            decision_message=decision_message,
            decision_color=decision_color,
            max_loan=loan_rec['recommended_max_loan'],
            monthly_capacity=loan_rec['monthly_payment_capacity'],
            estimated_payment=loan_rec['monthly_payment_capacity'] * 0.9,
            base_rate=5.0,
            adjusted_rate=5.0 + risk_info['interest_rate_adjustment'],
            annual_income=annual_income,
            credit_score=int(credit_score),
            timestamp=timestamp.strftime('%Y-%m-%d %H:%M:%S')
        )
        
    except Exception as e:
        return f"""
        <html>
        <body style="font-family: Arial; padding: 50px; text-align: center;">
            <h1 style="color: #dc3545;">❌ Error</h1>
            <p style="font-size: 1.2em; color: #666;">An error occurred during assessment:</p>
            <p style="color: #dc3545; font-weight: bold;">{str(e)}</p>
            <br>
            <a href="/" style="padding: 15px 30px; background: #667eea; color: white; text-decoration: none; border-radius: 50px;">
                ← Go Back
            </a>
        </body>
        </html>
        """

@app.route('/health')
def health():
    """API health check"""
    conn = get_db_connection()
    db_status = "connected" if conn else "disconnected"
    if conn:
        conn.close()
    
    return jsonify({
        'status': 'healthy',
        'model_version': config['model_version'],
        'database': db_status,
        'timestamp': datetime.now().isoformat()
    })

# if __name__ == '__main__':
#     print("="*70)
#     print("LOAN RISK ASSESSMENT WEB APPLICATION")
#     print("="*70)
#     print(f"Model version: {config['model_version']}")
#     print("="*70)
#     print("\n🌐 Opening web interface...")
#     print("Open your browser and go to: http://127.0.0.1:5000")
#     print("\n✨ Features:")
#     print("  • Fill in customer information in the web form")
#     print("  • Get instant risk assessment")
#     print("  • View loan recommendations")
#     print("  • See approval decision")
#     print("="*70)
    
#     app.run(host='0.0.0.0', port=5000, debug=True)
    
    if __name__ == '__main__':
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)