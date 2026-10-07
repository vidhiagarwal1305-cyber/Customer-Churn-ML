import pandas as pd
import matplotlib.pyplot as plt


# Load modeling data
data = pd.read_excel(
    "data/telco x modeling data.xlsx"
)

# Keep only records where data is known
data = data[data["data_known"] == "yes"].copy()

# Count Active and Churned customers
churn_counts = data["churn"].value_counts()

# Create bar chart
plt.figure(figsize=(7, 5))

churn_counts.plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")

plt.xticks(rotation=0)
plt.tight_layout()

# Display chart
plt.show()