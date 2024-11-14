# Naive Bayes Classifier for Play Tennis Dataset

This repository contains an implementation of a Naive Bayes Classifier for the "Play Tennis" dataset. The classifier is implemented from scratch in Python, without using machine learning libraries like scikit-learn. The code supports training, prediction, and evaluation of the model with both training and test datasets.

## Features

- Trains a Naive Bayes classifier on the "Play Tennis" dataset.
- Calculates prior probabilities and likelihoods with Laplace smoothing.
- Supports classification of a single test instance.
- Evaluates model accuracy on both training and test data.
- Outputs detailed logs for predictions.

## Prerequisites

- Python 3.x
- Required libraries:
  - `pandas`
  - `json`

You can install any missing libraries using:
pip install pandas

## Project Structure
`main.py`: Python file that has the script  <br> 
`play_tennis_data.json`: JSON file containing the "Play Tennis" dataset for training. <br> 
`test_data.json`: JSON file containing additional test instances. <br> 
`naive_bayes_model.json`: JSON file where the trained model (priors and likelihoods) is saved. <br> 
`classification_log.txt`: Log file generated during model evaluation on the training dataset. <br> 
`testing_log.txt`: Log file generated during model evaluation on the test dataset. <br> 
`README.md`: Documentation for using the code.

## Usage
Load the Data Ensure that play_tennis_data.json (training data) and test_data.json (test data) are in the same directory as the script.

### Run the Classifier

Run the code with:
```bash
python main.py
```

This will:

Load the training data and train the classifier.
Save the trained model to naive_bayes_model.json.
Evaluate the model on the training data and output accuracy.
If test data is provided in test_data.json, it will evaluate on test data and save the results to testing_log.txt.

## Testing a Single Instance

In the __main__ section of the script, a sample test instance is provided:

test_instance = {
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "High",
    "Wind": "Strong"
}

The classifier will predict the class for this instance and display the result in the terminal.

## Evaluation Logs
After training, the model’s performance on the training data will be recorded in classification_log.txt.
If test data is provided, its performance on this test data will be recorded in testing_log.txt.


## Files and Logs
`naive_bayes_model.json`: Stores the trained model (priors and likelihoods) for future predictions. <br> 
`classification_log.txt`: Logs the predicted vs actual results and accuracy on the training dataset. <br> 
`testing_log.txt`: Logs the predicted vs actual results and accuracy on the test dataset.

## Notes
To modify the test instance, update the dictionary in the __main__ section.
To improve model accuracy, additional data and feature engineering would be beneficial.
