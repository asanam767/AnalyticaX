import pandas as pd

# Load the test data
test_df = pd.read_csv('C:\\Hack\\test_set_features.csv')

# Assuming you have trained a model named 'model' and made predictions
# Replace 'predicted_probabilities_h1n1' and 'predicted_probabilities_seasonal' with your actual predicted probabilities
predicted_probabilities_h1n1 = model.predict_proba(test_df.drop(columns=['respondent_id']))[:, 1]
predicted_probabilities_seasonal = model.predict_proba(test_df.drop(columns=['respondent_id']))[:, 1]

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'respondent_id': test_df['respondent_id'],
    'h1n1_vaccine': predicted_probabilities_h1n1,
    'seasonal_vaccine': predicted_probabilities_seasonal
})

# Ensure the predicted probabilities are between 0.0 and 1.0
submission_df['h1n1_vaccine'] = submission_df['h1n1_vaccine'].clip(0.0, 1.0)
submission_df['seasonal_vaccine'] = submission_df['seasonal_vaccine'].clip(0.0, 1.0)

# Save the DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)
