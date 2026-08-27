# Car Price Prediction System

This project develops a machine learning system for predicting the selling price of used cars based on vehicle specifications and listing information.

The project includes a Jupyter notebook covering data inspection, data cleaning, exploratory data analysis (EDA), feature preprocessing, model training, model comparison, feature importance analysis, and prediction. A trained XGBoost model and the required preprocessing components are saved as `.pkl` files for later use.

A Dockerized Dash web application is also provided to allow users to enter vehicle information and obtain a predicted selling price through a web interface.

## Notebook

- `Car_Price_Prediction.ipynb`: Contains the complete machine learning workflow, including:
  - Dataset loading and initial inspection
  - Data cleaning
  - Exploratory Data Analysis (EDA)
  - Feature selection and one-hot encoding
  - Train-test splitting
  - Missing-value imputation
  - Feature scaling
  - Linear Regression baseline modeling
  - Comparison of Random Forest, XGBoost, Gradient Boosting, Ridge, and Lasso Regression
  - XGBoost feature importance analysis
  - Model saving and prediction using new input data

The trained XGBoost model, scaler, and model feature columns are saved as `.pkl` files for use by the web application.

## Web App

The web application is developed using Dash and provides a user-friendly interface for predicting used-car selling prices.

The application loads the saved XGBoost model and preprocessing components and applies the same feature-processing steps used during model development before generating predictions.

### Run Locally

1. Install the required dependencies:

   ```bash
   pip install -r app/requirements.txt