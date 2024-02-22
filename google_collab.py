import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
features_df = pd.read_csv('C:\\Hack\\training_set_features.csv')
labels_df = pd.read_csv('C:\\Hack\\training_set_labels.csv')

# Check for missing values
print("\nMissing Values in Training Set Features:")
print(features_df.isnull().sum())

# Handle missing values
features_df.ffill(inplace=True)  # Forward fill missing values

# Summary Statistics
missing_values = {task and objectives:
"The task involves analyzing data from the 2009 National H1N1 Flu Survey conducted by the CDC to understand the factors influencing individuals' likelihood of receiving the H1N1 and seasonal flu vaccines. The primary objective is to identify key predictors of vaccine uptake to inform public health strategies and vaccination campaigns. Through exploratory data analysis and predictive modeling, we aim to uncover insights that can aid in promoting vaccine acceptance and uptake, ultimately contributing to improved public health outcomes."
task and objectives:
"The task involves analyzing data from the 2009 National H1N1 Flu Survey conducted by the CDC to understand the factors influencing individuals' likelihood of receiving the H1N1 and seasonal flu vaccines. The primary objective is to identify key predictors of vaccine uptake to inform public health strategies and vaccination campaigns. Through exploratory data analysis and predictive modeling, we aim to uncover insights that can aid in promoting vaccine acceptance and uptake, ultimately contributing to improved public health outcomes."
