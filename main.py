import json
import numpy as np
import pandas as pd
from collections import defaultdict


# Dataset Preparation
def load_train_data():
    with open("play_tennis_data.json", "r") as file:
        data = json.load(file)
    return pd.DataFrame(data)

def load_test_data():
    with open("test_data.json", "r") as file:
        test_data = json.load(file)
    return pd.DataFrame(test_data)


class NaiveBayesClassifier:
    def __init__(self):
        self.model = {"priors": {}, "likelihoods": {}}
        self.classes = []
        self.feature_values = {}

    def train(self, data):
        # Identify classes and feature values
        self.classes = data["PlayTennis"].unique()
        features = data.columns.drop(["Day", "PlayTennis"])

        # Calculate prior probabilities
        class_counts = data["PlayTennis"].value_counts()
        total_count = len(data)
        self.model["priors"] = {cls: class_counts[cls] / total_count for cls in self.classes}

        # Calculate likelihoods with Laplace smoothing
        for feature in features:
            self.model["likelihoods"][feature] = {}
            self.feature_values[feature] = data[feature].unique()

            for cls in self.classes:
                cls_data = data[data["PlayTennis"] == cls]
                cls_count = class_counts[cls]
                self.model["likelihoods"][feature][cls] = {}

                for value in self.feature_values[feature]:
                    count = cls_data[cls_data[feature] == value].shape[0]
                    # Laplace smoothing
                    likelihood = (count + 1) / (cls_count + len(self.feature_values[feature]))
                    self.model["likelihoods"][feature][cls][value] = likelihood

        # Save the model to JSON file
        with open("naive_bayes_model.json", "w") as file:
            json.dump(self.model, file)


    def predict(self, instance):
        # Load the trained model from JSON file
        with open("naive_bayes_model.json", "r") as file:
            self.model = json.load(file)

        max_posterior = -1
        best_class = None
        log_details = []

        # Calculate posterior probabilities for each class
        for cls in self.model["priors"]:
            posterior = self.model["priors"][cls]  # Start with the prior probability
            log_details.append(f"Prior for class {cls}: {posterior}")

            for feature, value in instance.items():
                # Get likelihood of feature value given class
                if feature in self.model["likelihoods"]:
                    likelihood = self.model["likelihoods"][feature][cls].get(
                        value, 1 / (len(self.model["priors"]) + len(self.model["likelihoods"][feature][cls]))
                    )
                    posterior *= likelihood
                    log_details.append(f"P({value}|{cls}) = {likelihood}")

            log_details.append(f"Posterior for class {cls}: {posterior}")
            if posterior > max_posterior:
                max_posterior = posterior
                best_class = cls

        log_details.append(f"Predicted class: {best_class}")
        print("\n".join(log_details))  # Print details of the calculation
        return best_class

    def evaluate(self, data, log):
        correct = 0
        log_file = open(f"{log}.txt", "w")

        for i in range(len(data)):
            test_instance = data.iloc[i].drop(["Day", "PlayTennis"]).to_dict()
            actual_class = data.iloc[i]["PlayTennis"]
            predicted_class = classifier.predict(test_instance)

            log_file.write(f"Instance {i + 1} - Predicted: {predicted_class}, Actual: {actual_class}\n")
            if predicted_class == actual_class:
                correct += 1

        accuracy = correct / len(data)
        log_file.write(f"Accuracy: {accuracy}")
        log_file.close()
        return accuracy


# Main execution
if __name__ == "__main__":
    data = load_train_data()
    test_data = load_test_data()
    classifier = NaiveBayesClassifier()
    classifier.train(data)
    accuracy = classifier.evaluate(data,"classification_log")
    print(f"Model accuracy: {accuracy}\n")

    if not test_data.size == 0:
        print(f"---------Test Data---------\n")
        accuracy = classifier.evaluate(test_data, "testing_log")
        print(f"Test Accuracy: {accuracy}\n")
    else:
        print("No test data found\n")

    # Test a single instance
    test_instance = {
        "Outlook": "Sunny",
        "Temperature": "Cool",
        "Humidity": "High",
        "Wind": "Strong"
    }
    print(f"---------Test Instance---------\n")
    prediction = classifier.predict(test_instance)
    print(f"Predicted class for the test instance: {prediction}")
