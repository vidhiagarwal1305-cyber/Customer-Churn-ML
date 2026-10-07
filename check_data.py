import pandas as pd

# Load the historical/modeling dataset
modeling_data = pd.read_excel(
    "data/telco x modeling data.xlsx"
)

# Load the deployment/current customer dataset
deployment_data = pd.read_excel(
    "data/telco x deployment data.xlsx"
)

print("MODELING DATA")
print("=" * 50)
print(modeling_data.head())
print("\nShape:", modeling_data.shape)
print("\nColumns:")
print(modeling_data.columns.tolist())

print("\n\nDEPLOYMENT DATA")
print("=" * 50)
print(deployment_data.head())
print("\nShape:", deployment_data.shape)
print("\nColumns:")
print(deployment_data.columns.tolist())