import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Initialize FastAPI app
app = FastAPI()

# Define the path to the model artifact
MODEL_PATH = "model.joblib"

# Load the model globally when the app starts
model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit(1) # Exit if model is not found, as app cannot function

# Define input data model for Pydantic
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Route for the home page
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Iris Prediction API! Use /predict to get predictions."}

# Route for prediction
@app.post("/predict")
async def predict_iris(data: IrisFeatures):
    try:
        # Convert input data to a pandas DataFrame
        # Ensure the order of columns matches the training data
        input_df = pd.DataFrame([data.dict()])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # You might want to map the numerical prediction to a species name
        # Assuming your model outputs 0, 1, 2 for setosa, versicolor, virginica
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        predicted_class = species_map.get(prediction, "Unknown")

        return {"predicted_class": predicted_class}
    except Exception as e:
        return {"error": str(e), "message": "Prediction failed. Please check input data."}

# To run the app directly (for local testing without uvicorn command)
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
