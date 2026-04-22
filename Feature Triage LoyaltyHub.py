# =========================================================
# ECA STAGE: Feature Triage (Dual-Layer: Data-Informed + Constraint Enforcement) — LoyaltyHub Adaptation
# =========================================================
# Purpose:
# Construct a temporally valid, structurally disciplined, and data-informed feature space for churn analysis in LoyaltyHub
# combining data-driven signal awareness (Pre-Triage Layer) and strict epistemic enforcement (Formal Triage Layer).
#
# Dataset/Input:
# File path: "C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub\LoyaltyHub_framed.csv"
#
# Output:
# Primary: Feature Registry Table (feature metadata including temporal admissibility, structural classification,
# dynamic/static flag, feature type, priority, audit log)
# Secondary: Pre-Triage Profile Table, lists of pre-decision admissible features, diagnostic-only features,
# high-priority feature subset, churn-only dataset, audit log of decisions
# =========================================================

# ---------------------------------------------------------
# STAGE OBJECTIVES
# ---------------------------------------------------------

# Objective 0: Perform data-informed exploratory scan (Pre-Triage Layer)
# Objective 1: Apply Temporal Admissibility Gate and assign labels
# Objective 2: Enforce structural segmentation constraint
# Objective 3: Classify features into strict feature types
# Objective 4: Evaluate admissible features for exploratory priority

# ---------------------------------------------------------
# LOAD DEPENDENCIES
# ---------------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import skew

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

data_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub\LoyaltyHub_framed.csv"
df = pd.read_csv(data_path)

# Initialize metadata tables
feature_registry = pd.DataFrame(columns=[
    "feature_name", "temporal_admissibility", "structural_class",
    "is_dynamic", "feature_type", "priority", "audit_log"
])
pre_triage_profile = pd.DataFrame(columns=[
    "feature_name", "variance", "missing_rate", "cardinality", "skew", "flag"
])

# Dictionary to store statistics for data-driven segmentation
feature_stats = {}

# =========================================================
# OBJECTIVE 0: Pre-Triage Scan
# =========================================================

print("Executing Objective 0: Pre-Triage Scan...")

for col in df.columns:
    if col in ["Unnamed: 0", "churn", "churn_risk_score", "observation_cutoff", "temporal_flag", "duplicate_flag"]:
        continue
    
    series = df[col]
    
    # Compute variance safely for numeric columns only
    if pd.api.types.is_numeric_dtype(series):
        variance = series.var(skipna=True)
        skewness = skew(series.dropna())
    else:
        variance = np.nan
        skewness = np.nan
    
    missing_rate = series.isnull().mean()
    cardinality = series.nunique()
    
    flag = []
    if pd.notnull(variance) and variance == 0:
        flag.append("zero_variance")
    if missing_rate > 0.3:
        flag.append("high_missing")
    if cardinality <= 1:
        flag.append("low_cardinality")
    
    # Store feature stats in dictionary for later use
    feature_stats[col] = {
        "variance": variance,
        "missing_rate": missing_rate,
        "cardinality": cardinality,
        "skew": skewness,
        "flag": flag if flag else ["ok"]
    }
    
    pre_triage_profile.loc[len(pre_triage_profile)] = [
        col, variance, missing_rate, cardinality, skewness, ", ".join(flag) if flag else "ok"
    ]

# =========================================================
# OBJECTIVE 1: Temporal Admissibility
# =========================================================

print("Executing Objective 1: Temporal Admissibility Gate...")

temporal_admissibility = {}

for col in df.columns:
    if col.startswith("avg_") or col in ["joining_date", "last_visit_time", "days_since_last_login"]:
        if col in ["last_visit_time", "days_since_last_login"]:
            temporal_admissibility[col] = "Outcome-relative"
        else:
            temporal_admissibility[col] = "Ambiguous"
    elif col in ["points_in_wallet", "feedback", "complaint_status"]:
        temporal_admissibility[col] = "Pre-decision admissible"
    else:
        temporal_admissibility[col] = "Pre-decision admissible"

# =========================================================
# OBJECTIVE 2: Structural Classification
# =========================================================

print("Executing Objective 2: Structural Segmentation...")

structural_class = {}

behavioral_features = [
    col for col, stats in feature_stats.items()
    if ("points_in_wallet" in col or "feedback" in col or "complaint_status" in col
        or col.startswith("avg_") or col.startswith("used_") or col.startswith("offer_application")
        or col.startswith("past_"))
]

exposure_features = [col for col in df.columns if col.startswith("exposure")]

missingness_features = [col for col in df.columns if col.startswith("is_")]

for col in df.columns:
    if col in behavioral_features:
        structural_class[col] = "Behavioral"
    elif col in exposure_features:
        structural_class[col] = "Exposure"
    elif col in missingness_features:
        structural_class[col] = "Missingness"
    else:
        structural_class[col] = "Structural"

# =========================================================
# OBJECTIVE 3: Feature Type Classification
# =========================================================

print("Executing Objective 3: Feature Type Classification...")

feature_type = {}
is_dynamic = {}

for col in df.columns:
    series = df[col]
    
    is_dynamic[col] = True if structural_class.get(col) == "Behavioral" and col not in missingness_features else False
    
    if pd.api.types.is_numeric_dtype(series):
        feature_type[col] = "Continuous" if series.nunique() > 10 else "Discrete numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        feature_type[col] = "Time-based"
    else:
        feature_type[col] = "Categorical"

# =========================================================
# OBJECTIVE 4: Priority Evaluation
# =========================================================

print("Executing Objective 4: Priority Evaluation...")

priority = {}
audit_log = {}

for col in df.columns:
    if col in ["Unnamed: 0"]:
        continue
    
    adm = temporal_admissibility.get(col, "Pre-decision admissible")
    cls = structural_class.get(col, "Structural")
    dyn = is_dynamic.get(col, False)
    var = feature_stats[col]["variance"] if col in feature_stats else np.nan
    flags = feature_stats[col]["flag"] if col in feature_stats else []
    
    # Step 1: Hard constraints
    if adm == "Outcome-relative" or cls == "Exposure" or ("zero_variance" in flags):
        priority[col] = "Discard"
        audit_log[col] = "Discarded due to temporal, exposure, or zero variance constraint"
        continue
    
    # Step 2: Caps
    if adm == "Ambiguous":
        priority[col] = "Secondary"
        audit_log[col] = "Ambiguous temporal label → capped to Secondary"
        continue
    
    if not dyn:
        priority[col] = "Secondary"
        audit_log[col] = "Static feature → capped to Secondary"
        continue
    
    # Step 3: Behavioral qualification
    if cls == "Behavioral" and dyn and adm == "Pre-decision admissible":
        priority[col] = "High"
        audit_log[col] = "Behavioral, dynamic, pre-decision → High priority"
    else:
        priority[col] = "Secondary"
        audit_log[col] = "Defaulted to Secondary"

# =========================================================
# ASSEMBLE FEATURE REGISTRY
# =========================================================

for col in df.columns:
    if col in ["Unnamed: 0"]:
        continue
    feature_registry.loc[len(feature_registry)] = [
        col,
        temporal_admissibility.get(col, "Pre-decision admissible"),
        structural_class.get(col, "Structural"),
        is_dynamic.get(col, False),
        feature_type.get(col, "Unknown"),
        priority.get(col, "Secondary"),
        audit_log.get(col, "No issues")
    ]

# ---------------------------------------------------------
# STAGE OUTPUT
# ---------------------------------------------------------

feature_registry.to_csv("Feature_Registry_Table.csv", index=False)
pre_triage_profile.to_csv("Pre_Triage_Profile_Table.csv", index=False)

pre_decision_features = feature_registry.loc[
    feature_registry["temporal_admissibility"] == "Pre-decision admissible", "feature_name"
].tolist()
diagnostic_only_features = feature_registry.loc[
    feature_registry["temporal_admissibility"].isin(["Outcome-relative", "Ambiguous"]), "feature_name"
].tolist()
high_priority_features = feature_registry.loc[
    feature_registry["priority"] == "High", "feature_name"
].tolist()
churn_only_dataset = df[df["churn"] == 1]

pd.DataFrame(pre_decision_features, columns=["feature_name"]).to_csv("Pre_Decision_Features.csv", index=False)
pd.DataFrame(diagnostic_only_features, columns=["feature_name"]).to_csv("Diagnostic_Only_Features.csv", index=False)
pd.DataFrame(high_priority_features, columns=["feature_name"]).to_csv("High_Priority_Features.csv", index=False)
churn_only_dataset.to_csv("Churn_Only_Dataset.csv", index=False)

print("Stage outputs successfully saved.")

# =========================================================
# Feature Triage (Dual-Layer: Data-Informed + Constraint Enforcement) STAGE COMPLETE
# =========================================================

print("Feature Triage (Dual-Layer: Data-Informed + Constraint Enforcement) stage execution complete.")