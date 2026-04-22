# =========================================================
# ECA STAGE: Feature Engineering (A subset under target variable analysis)
# =========================================================
# Purpose:
# Gold + Silver subset becomes a clean behavioral sandbox where churn is
# non-deterministic, exposure-normalized, and measurement-aware.
#
# Dataset/Input:
# "C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub\LoyaltyHub_framed.csv"
#
# Output:
# CSV file with engineered features for Gold + Silver customers
# =========================================================

# ---------------------------------------------------------
# STAGE OBJECTIVES
# ---------------------------------------------------------
# 1. Remove Deterministic Groups
# 2. Isolate Behavioral Signal
# 3. Remove Exposure Bias
# 4. Prevent Structural Contamination
# 5. Encode Measurement System
# 6. Preserve Interpretability

# ---------------------------------------------------------
# LOAD DEPENDENCIES
# ---------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub\LoyaltyHub_framed.csv"
df = pd.read_csv(file_path)

# =========================================================
# OBJECTIVE 1: Remove Deterministic Groups
# =========================================================
print("Executing Objective 1: Remove Deterministic Groups...")

# Keep only Gold and Silver
valid_categories = ['Gold Membership', 'Silver Membership']
df_behavioral = df[df['membership_category'].isin(valid_categories)].copy()

# Sanity check
print("Remaining categories:", df_behavioral['membership_category'].unique())

# =========================================================
# OBJECTIVE 2: Define Exposure Proxy
# =========================================================
print("Executing Objective 2: Define Exposure Proxy...")

# Convert to datetime
df_behavioral['joining_date'] = pd.to_datetime(df_behavioral['joining_date'], errors='coerce')
df_behavioral['last_visit_time'] = pd.to_datetime(df_behavioral['last_visit_time'], errors='coerce')

# Compute exposure_days
df_behavioral['exposure_days'] = (df_behavioral['last_visit_time'] - df_behavioral['joining_date']).dt.days

# Remove invalid exposure
df_behavioral = df_behavioral[df_behavioral['exposure_days'].notna()]
df_behavioral = df_behavioral[df_behavioral['exposure_days'] >= 0]

print("Valid exposure rows:", len(df_behavioral))

# =========================================================
# OBJECTIVE 3: Residualize Exposure-Sensitive Features
# =========================================================
print("Executing Objective 3: Residualize Exposure-Sensitive Features...")

exposure_sensitive_features = ['points_in_wallet', 'feedback', 'complaint_status']
residual_models = {}

for feature in exposure_sensitive_features:
    if feature not in df_behavioral.columns:
        continue

    # Convert to numeric safely
    y = pd.to_numeric(df_behavioral[feature], errors='coerce')
    X = df_behavioral[['exposure_days']]

    # Drop rows where y is NaN
    valid_idx = y.notna()
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]

    # Skip if no valid samples
    if len(X_valid) == 0:
        print(f"Skipping residualization for {feature} (no valid data)")
        continue

    model = LinearRegression()
    model.fit(X_valid, y_valid)

    # Predict for all rows (fill NaN with 0 temporarily)
    y_full = y.fillna(0)
    expected = model.predict(X)

    # Compute residual
    df_behavioral[f'{feature}_residual'] = y_full - expected

    # Store model
    residual_models[feature] = model

# =========================================================
# OBJECTIVE 4: Prevent Structural Contamination
# =========================================================
print("Executing Objective 4: Prevent Structural Contamination...")

# Build residual feature list safely
residual_features = [
    f'{f}_residual'
    for f in exposure_sensitive_features
    if f'{f}_residual' in df_behavioral.columns
]

print("Residual features created:", residual_features)

# =========================================================
# OBJECTIVE 5: Encode Measurement System
# =========================================================
print("Executing Objective 5: Encode Measurement System...")

missing_features = ['points_in_wallet', 'region_category', 'preferred_offer_types']

for feature in missing_features:
    if feature in df_behavioral.columns:
        df_behavioral[f'is_{feature}_missing'] = df_behavioral[feature].isna().astype(int)

# =========================================================
# OBJECTIVE 6: Preserve Interpretability
# =========================================================
print("Executing Objective 6: Preserve Interpretability...")

# Drop leakage and administrative features safely
exclude_features = [
    'churn_risk_score',
    'security_no',
    'referral_id',
    'temporal_flag',
    'duplicate_flag',
    'observation_cutoff'
]

df_behavioral = df_behavioral.drop(
    columns=[col for col in exclude_features if col in df_behavioral.columns],
    errors='ignore'
)

# ---------------------------------------------------------
# STAGE OUTPUT
# ---------------------------------------------------------

output_path = file_path.replace("LoyaltyHub_framed.csv", "LoyaltyHub_engineered_gold_silver.csv")
df_behavioral.to_csv(output_path, index=False)

print("Stage outputs successfully saved.")

# =========================================================
# FEATURE ENGINEERING STAGE COMPLETE
# =========================================================

print("Feature Engineering stage execution complete.")