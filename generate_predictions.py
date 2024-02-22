import pandas as pd
import joblib

# Load the trained model
model = joblib.load('trained_model.pkl')

# Load the test data
test_df = pd.read_csv('C:\\Hack\\test_set_features.csv')

# Load the transformer object
transformer = joblib.load('transformer.pkl')

# Preprocess the test data
X_test = transformer.transform(test_df)

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
