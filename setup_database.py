import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "database": "ml_monitoring",
    "user": "ml_user",
    "password": "StrongPassword123",
    "port": 5434
}

def setup_database():
    """Create all necessary tables"""
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    print("Creating predictions table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            customer_id VARCHAR(100),
            timestamp TIMESTAMP NOT NULL,
            input_features JSONB NOT NULL,
            risk_score FLOAT NOT NULL,
            recommended_loan FLOAT NOT NULL,
            approval_decision VARCHAR(50),
            
            -- Actual outcomes (filled in later)
            actual_loan_approved BOOLEAN,
            actual_loan_amount FLOAT,
            actual_default BOOLEAN,
            days_to_default INTEGER,
            outcome_notes TEXT,
            outcome_timestamp TIMESTAMP,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    print("Creating indexes on predictions table...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_customer_id ON predictions(customer_id);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp);
        CREATE INDEX IF NOT EXISTS idx_approval_decision ON predictions(approval_decision);
    """)
    
    print("Creating drift_metrics table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drift_metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            metric_value FLOAT NOT NULL,
            chunk_start TIMESTAMP,
            chunk_end TIMESTAMP,
            details JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_drift_timestamp ON drift_metrics(timestamp);
        CREATE INDEX IF NOT EXISTS idx_drift_metric_name ON drift_metrics(metric_name);
    """)
    
    print("Creating model_metrics table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            metric_value FLOAT NOT NULL,
            period_start TIMESTAMP,
            period_end TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON model_metrics(timestamp);
        CREATE INDEX IF NOT EXISTS idx_metrics_name ON model_metrics(metric_name);
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print("\n✅ Database setup complete!")
    print("Tables created:")
    print("  - predictions (stores all model predictions)")
    print("  - drift_metrics (stores drift detection results)")
    print("  - model_metrics (stores performance metrics)")

if __name__ == "__main__":
    setup_database()