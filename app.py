from urllib import request

from fastapi import FastAPI,Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

#templates
templates = Jinja2Templates(directory="templates")

with open("ZGA-heart_disease.pkl", "rb") as f:
    saved_data = pickle.load(f)
    scaler = saved_data["scaler"]
    pca = saved_data["pca"]
    model = saved_data["model"]

prediction_labels = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}
prediction_explanations = {
    0: "High clinical risk detected. Significant ST depression (oldpeak) and reduced maximum heart rate suggest potential coronary stress or ischemia.",
    1: "Low clinical risk. Vital signs and EKG patterns are within healthy ranges, showing optimal heart rate performance and minimal cardiac stress.",
    2: "Medium risk. While EKG results are stable, advanced age or elevated cholesterol levels indicate a need for lifestyle monitoring and preventive care."
}


class HealthValues(BaseModel):
    age : int
    sex : int
    cp : int
    trestbps : int
    chol : int
    fbs : int
    restecg : int
    thalach : int
    exang : int
    oldpeak : float
    slope : int

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(features : HealthValues):
    input_data = pd.DataFrame([features.model_dump()])

    input_scaled = scaler.transform(input_data)

    input_pca = pca.transform(input_scaled)

    prediction = model.predict(input_pca)

    return {"cluster_prediction" : int(prediction[0])}