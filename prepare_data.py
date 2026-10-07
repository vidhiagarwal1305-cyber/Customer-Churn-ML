import pandas as pd

# Load modeling data
data = pd.read_excel(
    "data/telco x modeling data.xlsx"
)

# Keep only records where data is known
data = data[data["data_known"] == "yes"].copy()

# Select the variables required for the ML model
selected_columns = [
    "gender",
    "age",
    "tariff",
    "dropped_calls",
    "handset",
    "churn"
]

data = data[selected_columns]

# Remove rows with missing values
data = data.dropna()

print("Prepared Data")
print("=" * 50)

print("Shape:", data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

print("\nChurn distribution:")
print(data["churn"].value_counts())