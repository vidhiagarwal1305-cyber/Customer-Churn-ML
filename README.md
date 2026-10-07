# Customer Churn Prediction using Machine Learning

## Project Overview

This project focuses on predicting customer churn using Machine Learning.

The objective is to use historical Telco customer data to train a classification model and identify current customers who have a high probability of churning.

The project is implemented in Python using a Decision Tree classification model.

## Objective

The main objectives of this project are:

- Analyze historical customer data.
- Identify useful factors related to customer churn.
- Train a Machine Learning classification model.
- Evaluate the model's performance.
- Predict churn for current/deployment customers.
- Identify customers with a high probability of churning.

## Dataset

Two datasets are used in this project:

1. *Modeling Dataset*
   - Contains historical customer information.
   - Used to train and test the Machine Learning model.

2. *Deployment Dataset*
   - Contains current customer information.
   - Used to make churn predictions for customers whose churn status is unknown.

## Features Used

The following features are used to predict customer churn:

- Gender
- Age
- Tariff
- Dropped Calls
- Handset

### Target Variable

The target variable is:

- *Churn*

It indicates whether a customer is Active or Churned.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- OpenPyXL
- Joblib
- VS Code

## Machine Learning Model

A *Decision Tree Classifier* is used for customer churn prediction.

The categorical variables are converted into numerical form using One-Hot Encoding before being provided to the model.

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

## Model Performance

The Decision Tree model achieved an accuracy of:

*83.4%*

This means the model correctly classified approximately 83 out of every 100 customers in the test dataset.

## Customer Churn Prediction

After training the model, it was applied to the deployment dataset.

The model predicted the churn status and churn probability for each customer.

Customers predicted as *Churned* with a churn probability greater than 90% were identified as high-risk customers.

### Result

*3,197 customers* were identified as customers at risk of churn.

The results are saved in:

customers_at_risk.csv

The output contains:

- Customer ID
- Predicted Churn
- Churn Probability

## Project Workflow

```text
Historical Data
       ↓
Data Preparation
       ↓
Feature Selection
       ↓
Data Preprocessing
       ↓
Train-Test Split
       ↓
Decision Tree Model
       ↓
Model Evaluation
       ↓
Deployment Data
       ↓
Churn Prediction
       ↓
Identify High-Risk Customers
       ↓
customers_at_risk.csv