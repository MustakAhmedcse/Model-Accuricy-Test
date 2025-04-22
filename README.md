# Open AI Model Accuracy Test for Name Realism Classification

This project is designed to evaluate the performance and accuracy of multiple language models (`gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o-mini`) in classifying names as **Realistic** or **Not Realistic** through a local Flask API.

---

## 📁 Project Structure

```
.
├── app.py                    # Flask API serving predictions
├── test_models_parallel.py  # Parallel test script to evaluate models
├── test_names.csv           # CSV test dataset with Name and Is_Valid columns
├── model_comparison_results.json  # Output of evaluation results
└── README.md                # This documentation file
```

---

## 🧪 Dataset Format

The test dataset (`test_names.csv`) must have the following format:

```csv
Name,Is_Valid
John,1
Xyzzy,0
Alice,1
...
```

- `Name`: The name to test.
- `Is_Valid`: Binary label (`1` for Realistic, `0` for Not Realistic).

---

## 🚀 How to Run

### 1. Start the Flask API

```bash
python app.py
```

This starts a server at `http://localhost:5000/predict`.

### 2. Run the Test Script

```bash
python test_models_parallel.py
```

This will:
- Load the test dataset.
- Send names to the API in parallel per model.
- Measure accuracy and runtime for each model.
- Save results to `model_comparison_results.json`.

---

## 📊 Accuracy Comparison Table

| Model         | Phase        | Date       | Accuracy (%) | Time (s) |
|---------------|--------------|------------|--------------|----------|
| gpt-4.1       | Phase 1      | -          | 97.19        | 262.27   |
| gpt-4.1-mini  | Phase 1      | -          | 98.43        | 272.02   |
| gpt-4.1-nano  | Phase 1      | -          | 89.33        | 237.47   |
| gpt-4o-mini   | Phase 1      | -          | 96.29        | 247.08   |
| gpt-4.1       | Phase 2      | 2025-04-22 | 94.11        | 272.75   |
| gpt-4.1-mini  | Phase 2      | 2025-04-22 | 95.05        | 284.55   |
| gpt-4.1-nano  | Phase 2      | 2025-04-22 | 88.21        | 251.90   |
| gpt-4o-mini   | Phase 2      | 2025-04-22 | 92.53        | 266.34   |
| gpt-4.1       | Phase 2.1    | 2025-04-22 | 95.47        | 276.89   |
| gpt-4.1-mini  | Phase 2.1    | 2025-04-22 | 96.53        | 285.68   |
| gpt-4.1-nano  | Phase 2.1    | 2025-04-22 | 90.21        | 253.02   |
| gpt-4o-mini   | Phase 2.1    | 2025-04-22 | 94.42        | 266.13   |

---

## ⚙️ Dependencies

Install required libraries:

```bash
pip install -r requirements.txt
```

Minimal `requirements.txt`:

```
Flask
pandas
scikit-learn
requests
retrying
```

---

## 📌 Notes

- The `test_models_parallel.py` script is optimized using multiprocessing to send parallel requests per model.
- The API expects a JSON input like:

```json
{
  "name": "John",
  "model": "gpt-4.1"
}
```

And responds with:

```json
{
  "prediction": "Realistic"
}
```

---


