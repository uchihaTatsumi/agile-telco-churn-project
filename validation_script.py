import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Telco Customer Churn.csv")

# Apply the same preparation steps used before modelling
df_validated = df.copy()

# Convert TotalCharges into numeric format
df_validated["TotalCharges"] = pd.to_numeric(df_validated["TotalCharges"], errors="coerce")

# Handle missing values in TotalCharges
df_validated["TotalCharges"] = df_validated["TotalCharges"].fillna(0)

# Convert Churn into binary format
df_validated["Churn"] = df_validated["Churn"].map({"No": 0, "Yes": 1})

# Automated validation checks
validation_results = {}

# Check 1: Missing values
validation_results["Missing Values"] = df_validated.isnull().sum().sum() == 0

# Check 2: Duplicate records based on customerID
validation_results["Duplicate Customer IDs"] = df_validated["customerID"].duplicated().sum() == 0

# Check 3: TotalCharges data type
validation_results["TotalCharges Numeric"] = pd.api.types.is_numeric_dtype(df_validated["TotalCharges"])

# Check 4: Valid churn values
validation_results["Valid Churn Values"] = set(df_validated["Churn"].unique()).issubset({0, 1})

# Display validation results
for check, result in validation_results.items():
    status = "PASSED" if result else "FAILED"
    print(f"{check}: {status}")

# Final validation decision
if all(validation_results.values()):
    print("\nFinal Result: All validation checks passed. Dataset is ready for modelling.")
else:
    print("\nFinal Result: Some validation checks failed. Dataset requires further cleaning.")
