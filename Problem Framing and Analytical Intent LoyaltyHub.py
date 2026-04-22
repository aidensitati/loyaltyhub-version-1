# =========================================================  
# ECA STAGE: Data Framing and Analytical Intent  
# =========================================================  
# Purpose:  
# Establish a formally constrained analytical frame for examining churn within the LoyaltyHub dataset.  
# This includes defining a binary churn outcome, enforcing temporal alignment, validating account uniqueness, 
# and mapping features into behavioral and structural categories.  
#  
# Dataset/Input:  
# LoyaltyHub dataset with 23 columns: "C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub\LoyaltyHub.csv"  
# Columns: age, gender, security_no, region_category, membership_category, joining_date, joined_through_referral, 
# referral_id, preferred_offer_types, medium_of_operation, internet_option, last_visit_time, days_since_last_login, 
# avg_time_spent, avg_transaction_value, avg_frequency_login_days, points_in_wallet, used_special_discount, 
# offer_application_preference, past_complaint, complaint_status, feedback, churn_risk_score  
#  
# Output:  
# Primary: Binary churn outcome, temporally aligned dataset, unique account-level observations, feature classification mapping  
# Secondary: Feature mapping table, validation logs for temporal alignment and uniqueness  
# =========================================================  

# ---------------------------------------------------------  
# STAGE OBJECTIVES  
# ---------------------------------------------------------
# Objective 1: Convert churn_risk_score into a binary churn label using a fixed threshold or rule.
# Objective 2: Use joining_date and engagement features to establish temporal alignment for all observations.
# Objective 3: Validate uniqueness of security_no and enforce account-level unit of analysis.
# Objective 4: Map all features into behavioral and structural categories and create a feature mapping table.

# ---------------------------------------------------------  
# LOAD DEPENDENCIES  
# ---------------------------------------------------------
import pandas as pd
import numpy as np

# ---------------------------------------------------------  
# LOAD DATA  
# ---------------------------------------------------------
data_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub\LoyaltyHub.csv"
df = pd.read_csv(data_path)

# =========================================================  
# OBJECTIVE 1: Binary Churn Label Definition  
# =========================================================  
print("Executing Objective 1: Binary Churn Label Definition...")

# Define a binary churn variable
# Thresholding rule: churn_risk_score >= 0.5 is considered churned
df['churn'] = np.where(df['churn_risk_score'] >= 0.5, 1, 0)

# Confirm creation
print("Binary churn column created. Sample values:")
print(df[['churn_risk_score', 'churn']].head())

# =========================================================  
# OBJECTIVE 2: Temporal Alignment  
# =========================================================  
print("Executing Objective 2: Temporal Alignment...")

# Convert date columns to datetime
df['joining_date'] = pd.to_datetime(df['joining_date'], errors='coerce')
df['last_visit_time'] = pd.to_datetime(df['last_visit_time'], errors='coerce')

# Calculate last observable activity
df['observation_cutoff'] = df[['last_visit_time', 'joining_date']].max(axis=1)

# Flag any temporal inconsistencies (last visit before joining)
df['temporal_flag'] = df['last_visit_time'] < df['joining_date']

# =========================================================  
# OBJECTIVE 3: Account-Level Unit Validation  
# =========================================================  
print("Executing Objective 3: Account-Level Unit Validation...")

# Check for duplicate accounts
duplicate_accounts = df['security_no'].duplicated().sum()
df['duplicate_flag'] = df['security_no'].duplicated()

print(f"Duplicate accounts found: {duplicate_accounts}")

# Keep only first occurrence if duplicates exist (optional, depends on ECA policy)
df_unique = df.drop_duplicates(subset='security_no').copy()

# =========================================================  
# OBJECTIVE 4: Feature Mapping  
# =========================================================  
print("Executing Objective 4: Feature Mapping...")

# Behavioral features (pre-defined)
behavioral_features = [
    'last_visit_time', 'days_since_last_login', 'avg_time_spent', 'avg_transaction_value',
    'avg_frequency_login_days', 'points_in_wallet', 'used_special_discount',
    'offer_application_preference', 'past_complaint', 'complaint_status', 'feedback',
    'preferred_offer_types'
]

# Structural features (pre-defined)
structural_features = [
    'age', 'gender', 'region_category', 'security_no', 'membership_category',
    'joining_date', 'joined_through_referral', 'referral_id', 'medium_of_operation',
    'internet_option', 'churn_risk_score'
]

# Create mapping table
feature_mapping = pd.DataFrame({
    'feature_name': behavioral_features + structural_features,
    'feature_type': ['behavioral']*len(behavioral_features) + ['structural']*len(structural_features)
})

print("Feature mapping table created. Sample:")
print(feature_mapping.head())

# ---------------------------------------------------------  
# STAGE OUTPUT  
# ---------------------------------------------------------
# Primary dataset with churn label and validated accounts
primary_output_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub\LoyaltyHub_framed.csv"
df_unique.to_csv(primary_output_path, index=False)

# Feature mapping table
feature_mapping_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub\feature_mapping.csv"
feature_mapping.to_csv(feature_mapping_path, index=False)

# Validation log (temporal and duplicate flags)
validation_log = df_unique[['security_no', 'temporal_flag', 'duplicate_flag']]
validation_log_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub\validation_log.csv"
validation_log.to_csv(validation_log_path, index=False)

print("Stage outputs successfully saved.")

# =========================================================  
# Data Framing and Analytical Intent STAGE COMPLETE  
# =========================================================  
print("Data Framing and Analytical Intent stage execution complete.")