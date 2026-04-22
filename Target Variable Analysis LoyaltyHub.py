# =========================================================  
# ECA STAGE: Target Variable Analysis  
# =========================================================  
# Purpose:  
# To rigorously evaluate the churn target variable within the LoyaltyHub dataset by:
# • Quantifying class balance and segmentation structure
# • Characterizing temporal churn behavior under lifecycle constraints
# • Establishing baseline predictive performance bounds
# • Extracting target-driven metrics to guide disciplined feature engineering
#  
# Dataset/Input:  
# "C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub\LoyaltyHub_framed.csv"
#  
# Output:  
# churn_summary, segment_churn_rates, tenure_churn_distribution, baseline_performance  
# =========================================================  


# ---------------------------------------------------------  
# STAGE OBJECTIVES  
# ---------------------------------------------------------

# Objective 1:
# Compute overall churn rate and class distribution, including imbalance ratio.

# Objective 2:
# Compute churn rates across key customer segments (membership_category, region_category, preferred_offer_types) and quantify variance across segments.

# Objective 3:
# Construct lifecycle-based churn behavior by deriving tenure (last_visit_time - joining_date) and analyzing churn distribution across tenure bins.

# Objective 4:
# Establish baseline predictive performance using:
# • naive baseline (majority class)
# • simple heuristic rules based on top correlated non-leakage features

# Objective 5:
# Extract quantitative metrics (segment churn rates, temporal churn gradients, baseline errors) to inform future feature engineering.


# ---------------------------------------------------------  
# LOAD DEPENDENCIES  
# ---------------------------------------------------------

import pandas as pd
import numpy as np


# ---------------------------------------------------------  
# LOAD DATA  
# ---------------------------------------------------------

file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub\LoyaltyHub_framed.csv"
df = pd.read_csv(file_path)


# =========================================================  
# OBJECTIVE 1: Class Balance and Distribution  
# =========================================================  

print("Executing Objective 1: Class Balance and Distribution...")

total_count = len(df)
churned_count = df['churn'].sum()
retained_count = total_count - churned_count

churn_rate = df['churn'].mean()
imbalance_ratio = retained_count / churned_count if churned_count != 0 else np.nan

churn_summary = pd.DataFrame({
    'metric': ['total_count', 'churned_count', 'retained_count', 'churn_rate', 'imbalance_ratio'],
    'value': [total_count, churned_count, retained_count, churn_rate, imbalance_ratio]
})

print(churn_summary)


# =========================================================  
# OBJECTIVE 2: Segment-Level Churn Analysis  
# =========================================================  

print("Executing Objective 2: Segment-Level Churn Analysis...")

segment_features = ['membership_category', 'region_category', 'preferred_offer_types']
segment_results = []

for feature in segment_features:
    grouped = df.groupby(feature)['churn'].agg(['mean', 'count']).reset_index()
    grouped['feature'] = feature
    grouped.rename(columns={'mean': 'churn_rate', 'count': 'group_size'}, inplace=True)
    
    # Drop groups with missing values
    grouped = grouped.dropna()
    
    # Compute variance and range
    churn_variance = grouped['churn_rate'].var()
    churn_range = grouped['churn_rate'].max() - grouped['churn_rate'].min()
    
    grouped['variance'] = churn_variance
    grouped['range'] = churn_range
    
    segment_results.append(grouped)

segment_churn_rates = pd.concat(segment_results, ignore_index=True)

print(segment_churn_rates.head())


# =========================================================  
# OBJECTIVE 3: Temporal (Tenure-Based) Churn Behavior  
# =========================================================  

print("Executing Objective 3: Temporal Churn Behavior...")

# Convert to datetime
df['joining_date'] = pd.to_datetime(df['joining_date'], errors='coerce')
df['last_visit_time'] = pd.to_datetime(df['last_visit_time'], errors='coerce')

# Compute tenure
df['tenure_days'] = (df['last_visit_time'] - df['joining_date']).dt.days

# Remove invalid tenure values
df_valid_tenure = df[df['tenure_days'] >= 0].copy()

# Create tenure bins (quantiles)
df_valid_tenure['tenure_bin'] = pd.qcut(df_valid_tenure['tenure_days'], q=5, duplicates='drop')

# Compute churn per bin
tenure_summary = df_valid_tenure.groupby('tenure_bin')['churn'].agg(['mean', 'count']).reset_index()
tenure_summary.rename(columns={'mean': 'churn_rate', 'count': 'observations'}, inplace=True)

tenure_churn_distribution = tenure_summary

print(tenure_churn_distribution)


# =========================================================  
# OBJECTIVE 4: Baseline Predictive Performance  
# =========================================================  

print("Executing Objective 4: Baseline Predictive Performance...")

# Naive baseline (majority class)
majority_class = 0 if retained_count >= churned_count else 1
naive_predictions = np.full(shape=total_count, fill_value=majority_class)

naive_accuracy = (naive_predictions == df['churn']).mean()

# Heuristic baseline using median splits
heuristic_df = df.copy()

# Handle missing values
heuristic_df['points_in_wallet'] = heuristic_df['points_in_wallet'].fillna(heuristic_df['points_in_wallet'].median())

# Median thresholds
points_threshold = heuristic_df['points_in_wallet'].median()
transaction_threshold = heuristic_df['avg_transaction_value'].median()

# Simple rule-based prediction
heuristic_predictions = (
    (heuristic_df['points_in_wallet'] < points_threshold) &
    (heuristic_df['avg_transaction_value'] < transaction_threshold)
).astype(int)

heuristic_accuracy = (heuristic_predictions == heuristic_df['churn']).mean()

baseline_performance = pd.DataFrame({
    'model': ['naive', 'heuristic'],
    'accuracy': [naive_accuracy, heuristic_accuracy]
})

print(baseline_performance)


# =========================================================  
# OBJECTIVE 5: Metric Extraction for Feature Engineering  
# =========================================================  

print("Executing Objective 5: Metric Extraction...")

# Extract key metrics
feature_engineering_metrics = {
    'overall_churn_rate': churn_rate,
    'imbalance_ratio': imbalance_ratio,
    'avg_segment_variance': segment_churn_rates['variance'].mean(),
    'avg_segment_range': segment_churn_rates['range'].mean(),
    'avg_tenure_churn_rate': tenure_churn_distribution['churn_rate'].mean(),
    'naive_accuracy': naive_accuracy,
    'heuristic_accuracy': heuristic_accuracy
}

feature_engineering_metrics_df = pd.DataFrame(list(feature_engineering_metrics.items()), columns=['metric', 'value'])

print(feature_engineering_metrics_df)


# ---------------------------------------------------------  
# STAGE OUTPUT  
# ---------------------------------------------------------

output_path = r"C:/Users/hp/Exploratory Data Analysis/CSV files/LoyaltyHub"

churn_summary.to_csv(output_path + "churn_summary.csv", index=False)
segment_churn_rates.to_csv(output_path + "segment_churn_rates.csv", index=False)
tenure_churn_distribution.to_csv(output_path + "tenure_churn_distribution.csv", index=False)
baseline_performance.to_csv(output_path + "baseline_performance.csv", index=False)
feature_engineering_metrics_df.to_csv(output_path + "feature_engineering_metrics.csv", index=False)

print("Stage outputs successfully saved.")


# =========================================================  
# Target Variable Analysis STAGE COMPLETE  
# =========================================================  

print("Target Variable Analysis stage execution complete.")