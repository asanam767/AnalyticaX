import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load the saved model
model = joblib.load('trained_model.pkl')

# Load the test data
test_df = pd.read_csv('C:\\Hack\\test_set_features.csv')

# Preprocess the test data using the same preprocessing steps as in your training code
# Assuming X_train is defined and contains your training data
# Define and fit the ColumnTransformer on your training data
categorical_cols = test_df.select_dtypes(include=['object']).columns
one_hot_encoder = OneHotEncoder(drop='first')
ct = ColumnTransformer(transformers=[('encoder', one_hot_encoder, categorical_cols)], remainder='passthrough')

# Fit the ColumnTransformer on your training data
# Assuming X_train is defined and contains your training data
X_train = pd.read_csv('C:\\Hack\\training_set_features.csv')  # Load your training data
y_train = pd.read_csv('C:\\Hack\\training_set_labels.csv')  # Load your training labels
ct.fit(X_train)  # Fit the ColumnTransformer on your training data

# Transform the test data
X_test = ct.transform(test_df)

# Generate predictions for the test data
predictions = model.predict_proba(X_test)[:, 1]

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'respondent_id': test_df['respondent_id'],
    'h1n1_vaccine': predictions,
    'seasonal_vaccine': predictions
})

# Ensure the predicted probabilities are between 0.0 and 1.0
submission_df['h1n1_vaccine'] = submission_df['h1n1_vaccine'].clip(0.0, 1.0)
submission_df['seasonal_vaccine'] = submission_df['seasonal_vaccine'].clip(0.0, 1.0)

# Save the predictions to a CSV file
submission_df.to_csv('submission.csv', index=False)
