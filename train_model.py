import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load modeling data
data = pd.read_excel(
    "data/telco x modeling data.xlsx"
)

# Keep only records where data is known
data = data[data["data_known"] == "yes"].copy()

# Select required variables
features = [
    "gender",
    "age",
    "tariff",
    "dropped_calls",
    "handset"
]

target = "churn"

data = data[features + [target]].dropna()

X = data[features]
y = data[target]


# Identify categorical and numerical columns
categorical_features = [
    "gender",
    "tariff",
    "handset"
]

numerical_features = [
    "age",
    "dropped_calls"
]


# Convert categorical data into numbers
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# Create Decision Tree model
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)


# Combine preprocessing + model
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Train the model
pipeline.fit(X_train, y_train)


# Make predictions
y_pred = pipeline.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("MODEL TRAINING COMPLETED")
print("=" * 50)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save the trained model
joblib.dump(
    pipeline,
    "model/churn_model.pkl"
)

print("\nModel saved successfully!")
print("Location: model/churn_model.pkl")