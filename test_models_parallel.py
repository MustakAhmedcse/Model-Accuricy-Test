import pandas as pd
import requests
import json
import logging
import time
from retrying import retry
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask API endpoint
API_URL = "http://localhost:5000/predict"

# Models to test
MODELS = ["gpt-4.1-mini","gpt-4o-mini"]

# Configure concurrency
MAX_WORKERS = 20  # adjust based on your API rate limits

# Retry decorator for handling temporary API failures
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def send_prediction_request(name, model, session):
    """Send a prediction request to the Flask API using a shared session."""
    payload = {"name": name, "model": model}
    response = session.post(API_URL, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def load_dataset(file_path):
    """Load the test dataset from a CSV file."""
    df = pd.read_csv(file_path)
    if 'Name' not in df.columns or 'Is_Valid' not in df.columns:
        raise ValueError("CSV must contain 'Name' and 'Is_Valid' columns")
    return df


def evaluate_model(df, model):
    """Evaluate a single model on the dataset using parallel requests."""
    actuals = df['Is_Valid'].tolist()
    names = df['Name'].tolist()
    predictions = [None] * len(names)

    session = requests.Session()
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(send_prediction_request, name, model, session): idx
            for idx, name in enumerate(names)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                pred = result.get('prediction')
                predictions[idx] = 1 if pred == 'Realistic' else 0
            except Exception as e:
                logger.error(f"Error for name '{names[idx]}': {e}")
                predictions[idx] = 0  # default on error

    duration = time.time() - start_time
    accuracy = accuracy_score(actuals, predictions)
    logger.info(f"Model {model} completed in {duration:.2f}s with accuracy {accuracy:.4f}")
    return accuracy, predictions


def main():
    dataset_path = "test_names.csv"
    df = load_dataset(dataset_path)
    logger.info(f"Loaded dataset with {len(df)} names")

    results = {}
    for model in MODELS:
        accuracy, preds = evaluate_model(df, model)
        results[model] = {"accuracy": accuracy, "predictions": preds}

    print("\n=== Accuracy Comparison ===")
    for model, data in results.items():
        acc = data['accuracy']
        print(f"{model}: {acc:.4f} ({acc*100:.2f}%)")

    with open("model_comparison_results.json", "w") as f:
        detailed = {
            model: {
                "accuracy": res['accuracy'],
                "predictions": [
                    {"name": name, "actual": actual, "predicted": pred}
                    for name, actual, pred in zip(df['Name'], df['Is_Valid'], res['predictions'])
                ]
            }
            for model, res in results.items()
        }
        json.dump(detailed, f, indent=2)
    logger.info("Detailed results saved to model_comparison_results.json")


if __name__ == "__main__":
    main()
