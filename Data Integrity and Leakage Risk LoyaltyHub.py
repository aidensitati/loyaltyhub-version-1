# =========================================================  
# ECA STAGE: Data Integrity and Leakage Risk  
# =========================================================  
# Purpose:  
# To evaluate the epistemic validity of the LoyaltyHub dataset by distinguishing between:
# Information that is genuinely observable prior to churn
# Information that implicitly or explicitly encodes the outcome
#  
# Dataset/Input:  
# C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub\LoyaltyHub_framed.csv
#  
# Output:  
# Epistemic classification table, leakage flags, exposure summary, missingness report
# =========================================================


# ---------------------------------------------------------  
# STAGE OBJECTIVES  
# ---------------------------------------------------------
# Objective 1: Classify each feature based on temporal observability relative to churn (pre-outcome vs potentially post-outcome).
# Objective 2: Identify and flag leakage-prone variables by detecting features that strongly encode or depend on the churn outcome.
# Objective 3: Quantify exposure asymmetry between churned and retained users using lifecycle proxies (days_since_last_login, last_visit_time).
# Objective 4: Assess measurement consistency and missingness patterns across churn states for all features.
# Objective 5: Assign each variable to an epistemic category (pre-outcome observable, exposure-sensitive, lifecycle-terminal, retrospective, administrative).


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
# OBJECTIVE 1: Temporal Observability Classification  
# =========================================================  
print("Executing Objective 1: Temporal Observability Classification...")

# Convert date columns
df['joining_date'] = pd.to_datetime(df['joining_date'], errors='coerce')
df['last_visit_time'] = pd.to_datetime(df['last_visit_time'], errors='coerce')

temporal_observability = []

for col in df.columns:
    if col in ['last_visit_time', 'days_since_last_login',
               'avg_time_spent', 'avg_transaction_value',
               'avg_frequency_login_days']:
        temporal_observability.append((col, True))
    else:
        temporal_observability.append((col, False))

temporal_df = pd.DataFrame(temporal_observability, columns=['feature', 'pre_outcome_observable'])


# =========================================================  
# OBJECTIVE 2: Leakage Detection  
# =========================================================  
print("Executing Objective 2: Leakage Detection...")

leakage_results = []

for col in df.columns:
    if col == 'churn':
        continue
    try:
        if df[col].dtype in ['int64', 'float64']:
            if df[col].nunique() <= 1:
               corr = np.nan
            else:
               corr = df[col].corr(df['churn'])
        else:
            corr = df[col].astype('category').cat.codes.corr(df['churn'])
        leakage_flag = abs(corr) > 0.8  # threshold for strong dependency
        leakage_results.append((col, corr, leakage_flag))
    except:
        leakage_results.append((col, np.nan, False))

leakage_df = pd.DataFrame(leakage_results, columns=['feature', 'correlation_with_churn', 'leakage_flag'])


# =========================================================  
# OBJECTIVE 3: Exposure Asymmetry Analysis  
# =========================================================  
print("Executing Objective 3: Exposure Asymmetry Analysis...")

churned = df[df['churn'] == 1]
retained = df[df['churn'] == 0]

exposure_summary = pd.DataFrame({
    'metric': ['days_since_last_login_mean', 'days_since_last_login_std'],
    'churned': [
        churned['days_since_last_login'].mean(),
        churned['days_since_last_login'].std()
    ],
    'retained': [
        retained['days_since_last_login'].mean(),
        retained['days_since_last_login'].std()
    ]
})


# =========================================================  
# OBJECTIVE 4: Missingness Assessment  
# =========================================================  
print("Executing Objective 4: Missingness Assessment...")

missingness_data = []

for col in df.columns:
    missing_churned = churned[col].isnull().sum()
    missing_retained = retained[col].isnull().sum()
    missingness_data.append((col, missing_churned, missing_retained))

missingness_df = pd.DataFrame(missingness_data, columns=['feature', 'missing_churned', 'missing_retained'])


# =========================================================  
# OBJECTIVE 5: Epistemic Classification  
# =========================================================  
print("Executing Objective 5: Epistemic Classification...")

epistemic_classification = []

for col in df.columns:
    pre_obs = temporal_df.loc[temporal_df['feature'] == col, 'pre_outcome_observable'].values[0]
    leakage_match = leakage_df.loc[leakage_df['feature'] == col, 'leakage_flag']
    leakage_flag = leakage_match.values[0] if len(leakage_match) > 0 else False

    if leakage_flag:
        category = "administratively_deterministic"
    elif pre_obs:
        category = "pre_outcome_observable"
    elif col in ['days_since_last_login']:
        category = "exposure_sensitive"
    else:
        category = "retrospective_or_structural"

    epistemic_classification.append((col, category))

epistemic_df = pd.DataFrame(epistemic_classification, columns=['feature', 'epistemic_category'])


# ---------------------------------------------------------  
# STAGE OUTPUT  
# ---------------------------------------------------------
output_path = r"C:/Users/hp/Exploratory Data Analysis/CSV files/LoyaltyHub"

epistemic_df.to_csv(output_path + "epistemic_feature_classification.csv", index=False)
leakage_df.to_csv(output_path + "leakage_flags.csv", index=False)
exposure_summary.to_csv(output_path + "exposure_summary.csv", index=False)
missingness_df.to_csv(output_path + "missingness_report.csv", index=False)

print("Stage outputs successfully saved.")


# =========================================================  
# Data Integrity and Leakage Risk STAGE COMPLETE  
# =========================================================  
print("Data Integrity and Leakage Risk stage execution complete.")