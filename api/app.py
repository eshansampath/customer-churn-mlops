from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap

app = FastAPI()

model = joblib.load("models/model.pkl")
columns = joblib.load("models/columns.pkl")


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


# features
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str


@app.post("/predict")
def predict(data: CustomerData):
    input_df = pd.DataFrame([data.dict()])

    # Feature engineering
    input_df["CLV"] = input_df["MonthlyCharges"] * input_df["tenure"]

    # Encoding
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)
    input_df = input_df.astype(float)

    # Probability
    proba = model.predict_proba(input_df)[0][1]

    threshold = 0.65
    prediction = int(proba > threshold)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)

    shap_list = shap_values.values[0].tolist()

    return {
        "churn_prediction": prediction,
        "probability": float(proba),
        "shap_values": shap_list,
        "feature_names": list(input_df.columns)
    }