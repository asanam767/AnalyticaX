import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

# Load the datasets
features_df = pd.read_csv('C:\\Hack\\training_set_features.csv')
labels_df = pd.read_csv('C:\\Hack\\training_set_labels.csv')

# Check for missing values
print("\nMissing Values in Training Set Features:")
print(features_df.isnull().sum())

# Fill missing values
features_df = features_df.ffill()  # Use ffill to fill missing values

# Handle categorical variables
categorical_cols = features_df.select_dtypes(include=['object']).columns
one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
features_encoded = one_hot_encoder.fit_transform(features_df[categorical_cols])

# Concatenate encoded features with numerical features
features_df_encoded = pd.concat([features_df.drop(columns=categorical_cols), pd.DataFrame(features_encoded)], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_df_encoded, labels_df['h1n1_vaccine'], test_size=0.2, random_state=42)

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
sns.countplot(data=labels_df, x="h1n1_vaccine", palette="Set2")
plt.title("Distribution of H1N1 Vaccine")
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=labels_df, x="seasonal_vaccine", palette="Set2")
plt.title("Distribution of Seasonal Vaccine")
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(features_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()
