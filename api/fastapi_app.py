import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()
MODEL_PATH = "model.joblib" # Relative to where the app runs

model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit(1)

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Iris Prediction API! Use /predict to get predictions."}

@app.post("/predict")
async def predict_iris(data: IrisFeatures):
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)[0]
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        predicted_class = species_map.get(prediction, "Unknown")
        return {"predicted_class": predicted_class}
    except Exception as e:
        return {"error": str(e), "message": "Prediction failed. Please check input data."}
