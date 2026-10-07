import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model/churn_model.pkl")

# Load deployment data
data = pd.read_excel(
    "data/telco x deployment data.xlsx"
)

# Keep only records where data is known
data = data[data["data_known"] == "yes"].copy()

# Variables used by the model
features = [
    "gender",
    "age",
    "tariff",
    "dropped_calls",
    "handset"
]

# Remove rows with missing values
data = data.dropna(subset=features)

# Make churn predictions
data["predicted_churn"] = model.predict(data[features])

# Get probability of churn
data["churn_probability"] = model.predict_proba(
    data[features]
)[:, 1]

# Keep customers predicted to churn with probability > 90%
customers_at_risk = data[
    (data["predicted_churn"] == "Churned") &
    (data["churn_probability"] > 0.90)
].copy()

# Keep required output columns
customers_at_risk = customers_at_risk[
    ["customer_id", "predicted_churn", "churn_probability"]
]

# Save results
customers_at_risk.to_csv(
    "customers_at_risk.csv",
    index=False
)

print("PREDICTION COMPLETED")
print("=" * 50)

print("Deployment records used:", len(data))
print("Customers at risk:", len(customers_at_risk))

print("\nFirst 10 customers at risk:")
print(customers_at_risk.head(10))

print("\nResults saved as:")
print("customers_at_risk.csv")