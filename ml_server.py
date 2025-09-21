from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Load trained model
model = joblib.load("dropout_model.pkl")

# Define request format
class StudentData(BaseModel):
    attendance: float
    avg_marks: float
    fee_pending: float

# This is the FastAPI app object Uvicorn needs
app = FastAPI()

@app.post("/predict")
def predict(data: StudentData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0].tolist()
    return {
        "risk": int(prediction),
        "probabilities": {
            "continue": probability[0],
            "dropout": probability[1]
        }
    }
