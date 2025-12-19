import requests
import json

# API endpoint
url = "http://127.0.0.1:5000/predict"

# Sample customer data (use realistic values from your dataset)
test_customer = {
    "customer_id": "CUST_TEST_001",
    "features": {
        "Age": 35,
        "Experience": 10,
        "JobTenure": 3,
        "CreditScore": 680,
        "PaymentHistory": 0.95,
        "LengthOfCreditHistory": 8,
        "NumberOfOpenCreditLines": 3,
        "NumberOfCreditInquiries": 1,
        "PreviousLoanDefaults": 0,
        "BankruptcyHistory": 0,
        "UtilityBillsPaymentHistory": 1.0,
        "LoanDuration": 36,
        "BaseInterestRate": 4.5,
        "InterestRate": 5.2,
        "TotalDebtToIncomeRatio": 0.35,
        "MonthlyIncome_log": 10.5,
        "AnnualIncome_log": 11.0,
        "SavingsAccountBalance_log": 9.5,
        "CheckingAccountBalance_log": 8.2,
        "NetWorth_log": 11.5,
        "TotalAssets_log": 12.0,
        "TotalLiabilities_log": 10.8,
        "MonthlyLoanPayment_log": 7.5,
        "LoanAmount_log": 10.0,
        "MonthlyDebtPayments_log": 7.2,
        "EmploymentStatus": "Employed",
        "EducationLevel": "Bachelor",
        "MaritalStatus": "Married",
        "HomeOwnershipStatus": "Mortgage",
        "LoanPurpose": "Home"
    }
}

print("Sending prediction request...")
print("="*70)

try:
    response = requests.post(url, json=test_customer)
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n✅ PREDICTION SUCCESS!")
        print("="*70)
        print(f"\nCustomer ID: {result['customer_id']}")
        print(f"\n📊 RISK ASSESSMENT:")
        print(f"  Risk Score: {result['risk_assessment']['risk_score']:.4f}")
        print(f"  Risk Tier: {result['risk_assessment']['risk_tier']} ({result['risk_assessment']['risk_tier_code']})")
        print(f"  Description: {result['risk_assessment']['description']}")
        
        print(f"\n💰 LOAN RECOMMENDATION:")
        print(f"  Max Approved Amount: ${result['loan_recommendation']['max_approved_amount']:,.2f}")
        print(f"  Monthly Payment Capacity: ${result['loan_recommendation']['monthly_payment_capacity']:,.2f}")
        print(f"  Estimated Monthly Payment: ${result['loan_recommendation']['estimated_monthly_payment']:,.2f}")
        
        print(f"\n📋 LENDING TERMS:")
        print(f"  Base Interest Rate: {result['lending_terms']['base_interest_rate']:.2f}%")
        print(f"  Risk-Adjusted Rate: {result['lending_terms']['risk_adjusted_rate']:.2f}%")
        print(f"  Approval Decision: {result['lending_terms']['approval_decision']}")
        
        print(f"\n📝 METADATA:")
        print(f"  Annual Income: ${result['metadata']['annual_income']:,.2f}")
        print(f"  Credit Score: {result['metadata']['credit_score']}")
        print(f"  Prediction Time: {result['metadata']['prediction_timestamp']}")
        
        print("="*70)
        
        # Full JSON response
        print("\nFull JSON Response:")
        print(json.dumps(result, indent=2))
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Request failed: {e}")
    print("\nMake sure:")
    print("  1. The API is running (python api_service.py)")
    print("  2. You can access http://127.0.0.1:5000 in your browser")