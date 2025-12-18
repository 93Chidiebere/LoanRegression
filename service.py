"""
BentoML Service for Loan Risk Assessment & Recommendation System

This service provides REST API endpoints for:
1. Risk prediction and loan recommendations
2. Logging actual loan outcomes (for monitoring)
3. Health checks

Author: Your Name
Date: 2024
"""

import bentoml
import pandas as pd
import numpy as np
import pickle
import psycopg2
from datetime import datetime
import json
from typing import Dict, Any

# ============================================================================
# LOAD MODEL AND CONFIGURATION
# ============================================================================

# Load the trained pipeline
with open('risk_assessment_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Load deployment configuration
with open('deployment_config.pkl', 'rb') as f:
    config = pickle.load(f)

print("✓ Model and configuration loaded successfully")

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DB_CONFIG = {
    "host": "localhost",
    "database": "ml_monitoring",
    "user": "ml_user",
    "password": "your_password",  # Change this!
    "port": 5432
}

def get_db_connection():
    """Create database connection with error handling"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# ============================================================================
# HELPER FUNCTIONS (from notebook)
# ============================================================================

def recommend_loan_amount(
    risk_score: float,
    annual_income: float,
    existing_monthly_debt: float,
    credit_score: int,
    base_multiplier: float = 0.3,
    max_dti_ratio: float = 0.43
) -> Dict[str, Any]:
    """
    Recommend maximum safe loan amount based on risk assessment.
    """
    # Risk-adjusted income multiplier
    risk_adjusted_multiplier = base_multiplier * (1 - risk_score)
    
    # Credit score adjustment
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
    
    # Calculate available monthly payment capacity
    monthly_income = annual_income / 12
    max_total_monthly_debt = monthly_income * max_dti_ratio
    available_monthly_payment = max_total_monthly_debt - existing_monthly_debt
    
    if available_monthly_payment <= 0:
        return {
            'recommended_max_loan': 0,
            'monthly_payment_capacity': 0,
            'reason': 'Debt-to-income ratio already at maximum'
        }
    
    # Calculate max loan (5% interest, 5 years)
    monthly_rate = 0.05 / 12
    n_payments = 60
    
    max_loan_from_payment = available_monthly_payment * (
        ((1 + monthly_rate) ** n_payments - 1) / 
        (monthly_rate * (1 + monthly_rate) ** n_payments)
    )
    
    # Calculate max loan based on income multiple
    max_loan_from_income = annual_income * risk_adjusted_multiplier * credit_adjustment
    
    # Take minimum (most conservative)
    recommended_loan = min(max_loan_from_payment, max_loan_from_income)
    
    # Apply absolute cap
    absolute_max = annual_income * 0.5
    recommended_loan = min(recommended_loan, absolute_max)
    
    return {
        'recommended_max_loan': max(0, recommended_loan),
        'monthly_payment_capacity': available_monthly_payment,
        'risk_adjusted_multiplier': risk_adjusted_multiplier,
        'credit_adjustment': credit_adjustment
    }

def get_risk_tier_info(risk_score: float) -> Dict[str, Any]:
    """Categorize risk score into tiers"""
    if risk_score < 0.3:
        return {
            'tier': 'Low Risk',
            'tier_code': 'A',
            'interest_rate_adjustment': 0.0,
            'approval_recommendation': 'AUTO_APPROVE',
            'description': 'Excellent credit profile, minimal risk'
        }
    elif risk_score < 0.5:
        return {
            'tier': 'Medium-Low Risk',
            'tier_code': 'B',
            'interest_rate_adjustment': 1.0,
            'approval_recommendation': 'APPROVE',
            'description': 'Good credit profile, standard terms'
        }
    elif risk_score < 0.65:
        return {
            'tier': 'Medium Risk',
            'tier_code': 'C',
            'interest_rate_adjustment': 2.0,
            'approval_recommendation': 'MANUAL_REVIEW',
            'description': 'Acceptable risk with conditions'
        }
    elif risk_score < 0.8:
        return {
            'tier': 'Medium-High Risk',
            'tier_code': 'D',
            'interest_rate_adjustment': 3.5,
            'approval_recommendation': 'MANUAL_REVIEW_REQUIRED',
            'description': 'Elevated risk, requires careful review'
        }
    else:
        return {
            'tier': 'High Risk',
            'tier_code': 'E',
            'interest_rate_adjustment': 5.0,
            'approval_recommendation': 'DECLINE',
            'description': 'Significant risk factors present'
        }

def log_prediction_to_db(
    customer_id: str,
    input_features: Dict,
    risk_score: float,
    recommended_loan: float,
    approval_decision: str,
    timestamp: datetime
) -> bool:
    """Log prediction to database for monitoring"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions 
            (customer_id, timestamp, input_features, risk_score, 
             recommended_loan, approval_decision)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            customer_id,
            timestamp,
            json.dumps(input_features),
            float(risk_score),
            float(recommended_loan),
            approval_decision
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error logging to database: {e}")
        return False

# ============================================================================
# CREATE BENTOML SERVICE
# ============================================================================

svc = bentoml.Service("loan_risk_service")

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def predict(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction endpoint for risk assessment and loan recommendation.
    
    Expected input format:
    {
        "customer_id": "CUST123",
        "features": {
            "Age": 35,
            "CreditScore": 680,
            "AnnualIncome_log": 11.5,
            "MonthlyDebtPayments_log": 7.8,
            ... (all other features)
        }
    }
    
    Returns:
    {
        "customer_id": "CUST123",
        "risk_assessment": {...},
        "loan_recommendation": {...},
        "lending_terms": {...},
        "metadata": {...}
    }
    """
    try:
        # Extract customer ID and features
        customer_id = input_data.get('customer_id', 'UNKNOWN')
        features = input_data['features']
        
        # Convert to DataFrame (single row)
        customer_df = pd.DataFrame([features])
        
        # Ensure correct column order
        expected_features = config['feature_names']
        customer_df = customer_df[expected_features]
        
        # Predict risk score
        risk_score = pipeline.predict(customer_df)[0]
        
        # Get risk tier information
        risk_info = get_risk_tier_info(risk_score)
        
        # Extract financial data for loan recommendation
        annual_income = np.exp(features['AnnualIncome_log'])
        monthly_debt = np.exp(features['MonthlyDebtPayments_log'])
        credit_score = features['CreditScore']
        
        # Get loan recommendation
        loan_recommendation = recommend_loan_amount(
            risk_score=risk_score,
            annual_income=annual_income,
            existing_monthly_debt=monthly_debt,
            credit_score=credit_score
        )
        
        # Prepare response
        timestamp = datetime.now()
        
        response = {
            'customer_id': customer_id,
            'risk_assessment': {
                'risk_score': float(risk_score),
                'risk_tier': risk_info['tier'],
                'risk_tier_code': risk_info['tier_code'],
                'description': risk_info['description']
            },
            'loan_recommendation': {
                'max_approved_amount': float(loan_recommendation['recommended_max_loan']),
                'monthly_payment_capacity': float(loan_recommendation['monthly_payment_capacity']),
                'estimated_monthly_payment': float(loan_recommendation['monthly_payment_capacity'] * 0.9)
            },
            'lending_terms': {
                'base_interest_rate': 5.0,
                'risk_adjusted_rate': 5.0 + risk_info['interest_rate_adjustment'],
                'approval_decision': risk_info['approval_recommendation']
            },
            'metadata': {
                'model_version': config['model_version'],
                'prediction_timestamp': timestamp.isoformat(),
                'annual_income': float(annual_income),
                'credit_score': int(credit_score)
            }
        }
        
        # Log to database (async in production)
        log_prediction_to_db(
            customer_id=customer_id,
            input_features=features,
            risk_score=risk_score,
            recommended_loan=loan_recommendation['recommended_max_loan'],
            approval_decision=risk_info['approval_recommendation'],
            timestamp=timestamp
        )
        
        return response
        
    except Exception as e:
        return {
            'error': str(e),
            'message': 'Prediction failed. Please check input format.',
            'timestamp': datetime.now().isoformat()
        }

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def log_actual_outcome(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Endpoint to log actual loan outcomes for model monitoring.
    
    Expected input:
    {
        "customer_id": "CUST123",
        "loan_approved": true,
        "loan_amount": 25000,
        "actual_default": false,
        "days_to_default": null,
        "notes": "Loan fully repaid"
    }
    """
    try:
        conn = get_db_connection()
        if not conn:
            return {'error': 'Database connection failed'}
        
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE predictions
            SET 
                actual_loan_approved = %s,
                actual_loan_amount = %s,
                actual_default = %s,
                days_to_default = %s,
                outcome_notes = %s,
                outcome_timestamp = %s
            WHERE customer_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (
            input_data.get('loan_approved'),
            input_data.get('loan_amount'),
            input_data.get('actual_default'),
            input_data.get('days_to_default'),
            input_data.get('notes'),
            datetime.now(),
            input_data['customer_id']
        ))
        
        conn.commit()
        rows_affected = cursor.rowcount
        cursor.close()
        conn.close()
        
        return {
            'status': 'success',
            'message': f'Outcome logged for customer {input_data["customer_id"]}',
            'rows_updated': rows_affected,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def health_check(input_data: Dict = None) -> Dict[str, Any]:
    """
    Health check endpoint to verify service is running.
    """
    try:
        # Test database connection
        conn = get_db_connection()
        db_status = "connected" if conn else "disconnected"
        if conn:
            conn.close()
        
        return {
            'status': 'healthy',
            'model_version': config['model_version'],
            'database_status': db_status,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# ============================================================================
# MAIN (for testing locally)
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LOAN RISK ASSESSMENT SERVICE")
    print("="*70)
    print(f"Model version: {config['model_version']}")
    print(f"Training date: {config['training_date']}")
    print(f"Asymmetric costs: c_over={config['c_over']}, c_under={config['c_under']}")
    print("="*70)
    print("\nEndpoints available:")
    print("  - POST /predict : Risk assessment and loan recommendation")
    print("  - POST /log_actual_outcome : Log actual loan outcomes")
    print("  - POST /health_check : Service health check")
    print("="*70)