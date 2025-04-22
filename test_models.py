import pandas as pd
import requests
import json
import time
from sklearn.metrics import accuracy_score
import logging
from retrying import retry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask API endpoint
API_URL = "http://localhost:5000/predict"

# Models to test
#MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"]
MODELS = ["gpt-4o-mini"]

# Retry decorator for handling temporary API failures
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def send_prediction_request(name, model):
    """Send a prediction request to the Flask API."""
    payload = {
        "name": name,
        "model": model
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for name '{name}' with model '{model}': {str(e)}")
        raise

def load_dataset(file_path):
    """Load the test dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        if 'Name' not in df.columns or 'Is_Valid' not in df.columns:
            raise ValueError("CSV must contain 'Name' and 'Is_Valid' columns")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

def evaluate_model(df, model):
    """Evaluate a single model on the dataset."""
    predictions = []
    actuals = df['Is_Valid'].tolist()

    for name in df['Name']:
        try:
            logger.info(f"Testing name '{name}' with model '{model}'")
            result = send_prediction_request(name, model)
            
            if 'prediction' not in result:
                logger.warning(f"No prediction returned for name '{name}' with model '{model}': {result}")
                predictions.append(0)  # Default to incorrect
                continue

            prediction = result['prediction']
            # Convert prediction to binary (1 for Realistic, 0 for Not Realistic)
            predicted_value = 1 if prediction == "Realistic" else 0
            predictions.append(predicted_value)

            # Small delay to respect API rate limits
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error processing name '{name}' with model '{model}': {str(e)}")
            predictions.append(0)  # Default to incorrect on error

    accuracy = accuracy_score(actuals, predictions)
    return accuracy, predictions

def main():
    # Path to your test dataset
    dataset_path = "test_names.csv"

    try:
        # Load dataset
        df = load_dataset(dataset_path)
        logger.info(f"Loaded dataset with {len(df)} names")

        # Dictionary to store results
        results = {}

        # Evaluate each model
        for model in MODELS:
            logger.info(f"Evaluating model: {model}")
            accuracy, predictions = evaluate_model(df, model)
            results[model] = {
                "accuracy": accuracy,
                "predictions": predictions
            }
            logger.info(f"Model {model} accuracy: {accuracy:.4f}")

        # Print summary
        print("\n=== Accuracy Comparison ===")
        for model in MODELS:
            accuracy = results[model]["accuracy"]
            print(f"{model}: {accuracy:.4f} ({accuracy*100:.4f}%)")

        # Optionally, save detailed results to a file
        with open("model_comparison_results.json", "w") as f:
            detailed_results = {
                model: {
                    "accuracy": results[model]["accuracy"],
                    "predictions": [
                        {"name": name, "actual": actual, "predicted": pred}
                        for name, actual, pred in zip(df['Name'], df['Is_Valid'], results[model]["predictions"])
                    ]
                }
                for model in MODELS
            }
            json.dump(detailed_results, f, indent=2)
        logger.info("Detailed results saved to model_comparison_results.json")

    except Exception as e:
        logger.error(f"Test script failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()