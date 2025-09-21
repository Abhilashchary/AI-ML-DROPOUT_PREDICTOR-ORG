# AI-Based Student Dropout Prediction and Counseling System

## Project summary

This repository contains a lightweight end-to-end demo for predicting student dropout risk and presenting results in an interactive dashboard. It includes:

* Synthetic data generator (`create_data.py`) that creates a labeled `students.csv` dataset.
* Training script (`train_model.py`) which trains a `RandomForestClassifier` and saves it as `dropout_model.pkl`.
* A FastAPI model server (`ml_server.py`) exposing a `/predict` endpoint for online inference.
* A Streamlit dashboard (`app.py`) for uploading attendance/marks/fees Excel files, running predictions, visualizing risk, and re-training the model using uploaded data.
* A pre-trained `dropout_model.pkl` and `students.csv` (sample dataset).
* `requirements.txt` to reproduce the environment.

This is a demo/proof-of-concept that uses simple features and a simple rule for labeling during data generation. It is intended for demonstration and learning, not production use.

---

## Quick highlights / findings (analysis of the codebase you supplied)

* **Data generation** (`create_data.py`): creates 200 synthetic student records. Each student has `attendance`, `avg_marks`, and `fee_pending`. Labels (`dropout`) are generated using a **rule-based heuristic**:

  * dropout = 1 if `(attendance < 70) OR (avg_marks < 50) OR (fee_pending > 3000)` else 0

* **Training** (`train_model.py`): trains a `RandomForestClassifier` on `students.csv` using features `attendance`, `avg_marks` and `fee_pending` and target `dropout`. Model persisted with `joblib.dump` to `dropout_model.pkl`.

* **Serving** (`ml_server.py`): FastAPI app that loads `dropout_model.pkl` and exposes `POST /predict` which accepts JSON `{attendance, avg_marks, fee_pending}` and returns `{risk, probabilities: {continue, dropout}}`. The endpoint uses `model.predict` and `model.predict_proba`.

* **Dashboard** (`app.py`): Streamlit UI that:

  * Accepts three Excel uploads (attendance, marks, fees) with `student_id` and the expected feature column names (case-insensitive after stripping). Expected columns: `student_id`, `attendance`, `avg_marks`, `fee_pending`.
  * Preprocesses and merges inputs, applies the same rule-based label to mark `dropout` for newly uploaded rows.
  * Retrains a `RandomForestClassifier` (inside the app) combining existing `students.csv` and uploaded data; saves `dropout_model.pkl` and updates `students.csv`.
  * Predicts with the saved model and shows summary cards, bar/pie charts, detailed table, and a mailto generator for high-risk students.

* **Model file** `dropout_model.pkl` is present in repo. (Note: loading pickles across different numpy/scikit-learn versions can trigger import errors — use the `requirements.txt` or a matching environment.)

---

## Repository structure

```
AI-ML-DROPOUT_PREDICTOR-ORG-main/
├─ app.py                 # Streamlit dashboard (UI, retrain capability)
├─ create_data.py         # Synthetic data generator -> students.csv
├─ train_model.py         # Script to train model and save dropout_model.pkl
├─ ml_server.py           # FastAPI server exposing POST /predict
├─ dropout_model.pkl      # Saved trained model (joblib)
├─ students.csv           # Synthetic dataset (200 rows)
├─ requirements.txt       # Python deps
└─ README.md              # (original) brief readme
```

---

## Data (format & expectations)

### Expected CSV/Excel columns (case-insensitive after trimming)

* `student_id` — unique identifier for a student
* `attendance` — percentage or numeric attendance value (e.g. 85)
* `avg_marks` — numeric average marks (e.g. 70)
* `fee_pending` — numeric amount pending (e.g. 0 or 3000)

**Streamlit** uploads expect three Excel files (attendance, marks, fees). Each file must contain `student_id` and the relevant numeric column (attendance/avg\_marks/fee\_pending). The app normalizes column names to lowercase and trims whitespace.

---

## How the model is trained (details)

* **Data source**: `students.csv` (created by `create_data.py` or appended by the Streamlit app when retraining).
* **Features**: `attendance`, `avg_marks`, `fee_pending` (all numeric). No additional feature engineering or scaling in the provided scripts.
* **Target**: `dropout` (0 = continue, 1 = dropout). In the synthetic dataset, created by the rule: `(attendance < 70) OR (avg_marks < 50) OR (fee_pending > 3000)`.
* **Model**: `sklearn.ensemble.RandomForestClassifier` (defaults used in `train_model.py`; the Streamlit `retrain_model` sets `n_estimators=100, random_state=42`).
* **Training procedure**: `train_model.py` does an 80/20 train/test split but only fits the model and saves it; it does not print performance metrics. `app.py` retrains using all available combined data.
* **Persistence**: `joblib.dump(model, 'dropout_model.pkl')`.

**Note**: Because the labels are generated using a deterministic rule, a simple decision tree/random forest easily learns the pattern — accuracy will be artificially high. For a real problem, labels must be reliable (ground truth from institutional records) and more features considered.

---

## How the model is served

* The file `ml_server.py` defines a FastAPI app loading `dropout_model.pkl` at import time.
* Endpoint: `POST /predict`

  * **Request JSON**: \`{ "attendance": float, "avg\_marks": float, "fee\_pending": float }
  * **Response**: \`{ "risk": 0|1, "probabilities": { "continue": p0, "dropout": p1 } }

    * This maps `probabilities.continue` -> probability of class 0 and `probabilities.dropout` -> probability of class 1.
* Run with Uvicorn (example):

```bash
# from the project folder
uvicorn ml_server:app --reload --port 8000
```

* Example `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"attendance": 60, "avg_marks": 40, "fee_pending": 5000}'
```

**Important**: The server returns probabilities in the order the model's `classes_` attribute defines; in this code it is assumed `classes_ = [0,1]` — if that ever changed, the mapping should be updated to use `model.classes_` to map probabilities to class labels safely.

---

## How to run everything (step-by-step)

### 1) Prepare environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows (PowerShell)
pip install -r requirements.txt
```

### 2) (Optional) Generate sample data

```bash
python create_data.py
# this creates students.csv with 200 synthetic records
```

### 3) Train the model (produces dropout\_model.pkl)

```bash
python train_model.py
```

### 4) Start the model server

```bash
uvicorn ml_server:app --reload --port 8000
```

### 5) Start the Streamlit dashboard

```bash
streamlit run app.py
```

* Open the Streamlit UI (by default at [http://localhost:8501](http://localhost:8501)).
* Upload three Excel files (attendance, marks, fees) with the required columns. The app will merge, label data using the rule, retrain the model (combining `students.csv` and uploaded rows), save `dropout_model.pkl`, and show predictions & visualizations.

---

## API contract & examples

**POST /predict**

* Request:

```json
{
  "attendance": 59.0,
  "avg_marks": 40.0,
  "fee_pending": 5000.0
}
```

* Response:

```json
{
  "risk": 1,
  "probabilities": {
    "continue": 0.12,
    "dropout": 0.88
  }
}
```

---

## How predictions are made in the dashboard

* For every merged row the app either:

  * Uses the loaded model (`dropout_model.pkl`) to `predict` and `predict_proba`, or
  * Retrains a fresh RandomForest (if you use the retrain/merge flow) and then predicts.
* The dashboard also computes a local rule-based `dropout` field (same rule used to create the synthetic dataset) for labeling and retraining.

---

## Differences & small inconsistencies found in the repo

* `train_model.py` uses `RandomForestClassifier()` with default params, while `app.py`'s `retrain_model` uses `RandomForestClassifier(n_estimators=100, random_state=42)`.
* `ml_server.py` assumes `dropout_model.pkl` exists and is compatible with the environment. Loading a `joblib` pickle produced in a different `numpy`/`scikit-learn` version can raise import errors — see troubleshooting below.

---

## Troubleshooting & common gotchas

* **`ModuleNotFoundError` when loading `dropout_model.pkl` (e.g., `No module named 'numpy._core'`)**:

  * Likely caused by a mismatch between the environment used to create the pickle and the environment trying to load it.
  * Solution: recreate the model in your current environment (run `python train_model.py`) or ensure the exact same `numpy`/`scikit-learn` versions from `requirements.txt` are installed.

* **Pickle security**: Loading a pickle from an untrusted source is unsafe. Only load `dropout_model.pkl` you created yourself.

* **Column mismatch**: Uploaded Excel files must contain `student_id` and the expected numeric column. The UI normalizes column names to lowercase and strips whitespace.

* **Port conflicts**: If `uvicorn` or Streamlit fails to start due to ports, change ports explicitly.

* **CORS**: If hosting the API and frontend on different origins/hosts, enable CORS in the FastAPI app.

---

## Suggestions for improvements / next steps (roadmap)

1. Add proper **model evaluation** (accuracy, precision/recall, ROC AUC) and logging of metrics.
2. Improve **dataset realism** and include more features (demographics, semester-wise marks, attendance trend, engagement metrics).
3. Use **feature engineering** and **scaling** where appropriate. Consider model calibration for probability outputs.
4. Add **unit tests**, `pytest` coverage for prediction endpoints and data preprocessing.
5. Add a **Dockerfile** and `docker-compose.yml` to run the FastAPI server + Streamlit in containers.
6. Replace raw pickle usage with a **model registry** or versioned artifacts. Consider using `ONNX` or `PMML` for cross-platform inference.
7. Add **authentication** to the API and **role-based** access in the dashboard.
8. Add **explainability** (SHAP/LIME) output in the dashboard to show which features drive predictions per student.
9. Save historical predictions (and monitoring) for model drift detection.

---

## Security & ethical notes

* This demo uses synthetic data and a simple rule for labels. In real deployments, predictions about dropout risk are sensitive and may affect students' opportunities/advice — decisions must be made with caution and human oversight.
* Ensure data privacy (encrypt storage in transit/at rest), obtain necessary consent, and provide transparency for students/mentors.

---

## Reproducible commands summary

```bash
# create environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# generate data
python create_data.py

# (optional) train model
python train_model.py

# run API
uvicorn ml_server:app --reload --port 8000

# run dashboard
streamlit run app.py
```

---

## Contact / Further help

If you want, I can:

* Create a `Dockerfile` and `docker-compose.yml` to containerize the API + Streamlit app.
* Add unit tests for the API and preprocessing steps.
* Add a small evaluation script (compute accuracy, classification report) and visualize feature importance.

Tell me which of the above you'd like and I will create it for you.

---

## License

This repository is a demo. No license file provided — add one if you plan to share or use the project widely.
