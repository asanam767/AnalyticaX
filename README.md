# Flu Vaccine Prediction Project

## Overview
This project aims to predict the likelihood of individuals getting the H1N1 and seasonal flu vaccines based on demographic, behavioral, and health-related features. The predictions are made using machine learning models trained on data from the 2009 National H1N1 Flu Survey (NHFS) conducted by the CDC.

## Datasets
- `training_set_features.csv`: Contains features of individuals from the NHFS survey used for training the models.
- `training_set_labels.csv`: Contains labels indicating whether individuals received the H1N1 and seasonal flu vaccines.
- `test_set_features.csv`: Features of individuals from the NHFS survey for which predictions need to be made.

## Dependencies
- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn

## Setup
1. Clone this repository to your local machine.
2. Install the required dependencies using pip:

## Running the Code
1. Open a terminal or command prompt.
2. Navigate to the directory containing the project files.
3. Run the following command to execute the analysis:
This will perform exploratory data analysis, train machine learning models, and evaluate their performance.
4. To generate predictions for the test data and save them to a CSV file, run:
Ensure that the trained model is loaded and used for predictions.

## Results
The analysis includes exploratory data analysis, model training, evaluation, and prediction generation. The performance of the models is evaluated using the ROC AUC score.

## Future Work
- Experiment with different machine learning algorithms and hyperparameters.
- Explore feature engineering techniques to improve model performance.
- Consider ensembling multiple models for better predictions.

## Contributors
- [Katta Manasa](https://github.com/asanam767)

Feel free to contribute to this project by submitting pull requests or reporting issues.

