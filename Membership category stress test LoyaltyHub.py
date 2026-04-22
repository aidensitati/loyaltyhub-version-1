# =========================================================
# ECA STAGE: Membership Category Stress Test
# =========================================================
# Purpose:
# To rigorously evaluate whether membership_category behaves
# as a legitimate predictive feature or encodes structural/
# near-deterministic churn patterns.
#
# Dataset/Input:
# LoyaltyHub_framed.csv (with churn, temporal fields, validated structure)
#
# Output:
# Stress test reports for membership_category including:
# - churn distribution
# - group consistency
# - temporal dependence
# - cross-feature dependence
# =========================================================

# ---------------------------------------------------------
# LOAD DEPENDENCIES
# ---------------------------------------------------------
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv(r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub\LoyaltyHub_framed.csv")

# Drop index artifact if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Ensure datetime parsing
df['joining_date'] = pd.to_datetime(df['joining_date'], errors='coerce')
df['last_visit_time'] = pd.to_datetime(df['last_visit_time'], errors='coerce')

# =========================================================
# OBJECTIVE 1: Churn Distribution by Membership Category
# =========================================================
print("Executing Objective 1: Churn Distribution by Membership Category...")

membership_summary = df.groupby('membership_category').agg(
    churn_rate=('churn', 'mean'),
    churn_count=('churn', 'sum'),
    total_count=('churn', 'count')
).reset_index()

membership_summary['retained_count'] = membership_summary['total_count'] - membership_summary['churn_count']

print(membership_summary)

# =========================================================
# OBJECTIVE 2: Within-Group Consistency Check
# =========================================================
print("Executing Objective 2: Within-Group Consistency Check...")

# Check if any category is near-deterministic
membership_summary['is_near_deterministic'] = membership_summary['churn_rate'].apply(
    lambda x: True if (x >= 0.95 or x <= 0.05) else False
)

print(membership_summary[['membership_category', 'churn_rate', 'is_near_deterministic']])

# =========================================================
# OBJECTIVE 3: Temporal Dependence Check
# =========================================================
print("Executing Objective 3: Temporal Dependence Check...")

# Create tenure
df['tenure_days'] = (df['last_visit_time'] - df['joining_date']).dt.days

# Remove invalid tenure
df_valid_tenure = df[df['tenure_days'] >= 0].copy()

# Bin tenure
df_valid_tenure['tenure_bin'] = pd.qcut(df_valid_tenure['tenure_days'], q=5, duplicates='drop')

temporal_membership = df_valid_tenure.groupby(['membership_category', 'tenure_bin']).agg(
    churn_rate=('churn', 'mean'),
    observations=('churn', 'count')
).reset_index()

print(temporal_membership.head(20))

# =========================================================
# OBJECTIVE 4: Cross-Feature Dependency Check
# =========================================================
print("Executing Objective 4: Cross-Feature Dependency Check...")

# Compare membership with key features
cross_features = ['points_in_wallet', 'avg_transaction_value', 'avg_time_spent']

cross_feature_summary = []

for feature in cross_features:
    if feature in df.columns:
        temp = df.groupby('membership_category')[feature].mean().reset_index()
        temp['feature'] = feature
        cross_feature_summary.append(temp)

cross_feature_summary_df = pd.concat(cross_feature_summary, ignore_index=True)

print(cross_feature_summary_df.head(20))

# =========================================================
# OBJECTIVE 5: Stability Under Subsampling
# =========================================================
print("Executing Objective 5: Stability Under Subsampling...")

# Random subsample (50%)
df_sample = df.sample(frac=0.5, random_state=42)

sample_summary = df_sample.groupby('membership_category').agg(
    churn_rate=('churn', 'mean'),
    total_count=('churn', 'count')
).reset_index()

# Merge with original
stability_check = membership_summary.merge(
    sample_summary,
    on='membership_category',
    suffixes=('_full', '_sample')
)

stability_check['rate_diff'] = abs(
    stability_check['churn_rate_full'] - stability_check['churn_rate_sample']
)

print(stability_check)

# ---------------------------------------------------------
# STAGE OUTPUT
# ---------------------------------------------------------
membership_summary.to_csv("membership_churn_distribution.csv", index=False)
temporal_membership.to_csv("membership_temporal_behavior.csv", index=False)
cross_feature_summary_df.to_csv("membership_cross_feature_dependency.csv", index=False)
stability_check.to_csv("membership_stability_check.csv", index=False)

print("Membership category stress test outputs successfully saved.")

# =========================================================
# Membership Category Stress Test COMPLETE
# =========================================================
print("Membership Category Stress Test stage execution complete.")