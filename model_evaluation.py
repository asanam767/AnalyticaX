import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, ColumnTransformer  # Add this import statement

import joblib

# Load the datasets
features_df = pd.read_csv('C:\\Hack\\training_set_features.csv')
labels_df = pd.read_csv('C:\\Hack\\training_set_labels.csv')

# Check for missing values
print("\nMissing Values in Training Set Features:")
print(features_df.isnull().sum())

# Handle missing values
features_df.ffill(inplace=True)  # Forward fill missing values

# Handle categorical variables
categorical_cols = features_df.select_dtypes(include=['object']).columns
one_hot_encoder = OneHotEncoder(drop='first')
ct = ColumnTransformer(transformers=[('encoder', one_hot_encoder, categorical_cols)], remainder='passthrough')
X_encoded = ct.fit_transform(features_df)

# Save the transformer object
joblib.dump(ct, 'transformer.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, labels_df['h1n1_vaccine'], test_size=0.2, random_state=42)

# Train a machine learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict probabilities for the positive class
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate the ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC Score:", roc_auc)

# Visualize the target variables
plt.figure(figsize=(10, 5))
sns.countplot(data=labels_df, x="h1n1_vaccine", hue="h1n1_vaccine", palette="Set2", legend=False)
plt.title("Distribution of H1N1 Vaccine")
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=labels_df, x="seasonal_vaccine", hue="seasonal_vaccine", palette="Set2", legend=False)
plt.title("Distribution of Seasonal Vaccine")
plt.show()
