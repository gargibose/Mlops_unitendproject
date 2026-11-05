from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)

app = FastAPI(
    title="Titanic Survival Prediction API"
)

TEMPLATE_PATH = (
    Path(__file__).resolve().parent
    / "templates"
    / "index.html"
)

print("Template Path:", TEMPLATE_PATH)
print("Exists:", TEMPLATE_PATH.exists())


class PredictionRequest(BaseModel):

    Pclass: int
    Sex: str

    Age: Optional[float] = None
    SibSp: int = 0
    Parch: int = 0
    Fare: Optional[float] = None
    Embarked: Optional[str] = None


@app.get("/")
def home():

    if not TEMPLATE_PATH.exists():

        return {
            "error": "index.html not found",
            "path": str(TEMPLATE_PATH)
        }

    return FileResponse(TEMPLATE_PATH)


@app.post("/predict")
def predict(data: PredictionRequest):

    try:

        custom_data = CustomData(
            Pclass=data.Pclass,
            Sex=data.Sex,
            Age=data.Age,
            SibSp=data.SibSp,
            Parch=data.Parch,
            Fare=data.Fare,
            Embarked=data.Embarked,
        )

        pred_df = custom_data.get_data_as_dataframe()

        print("\nInput Data:")
        print(pred_df)

        predict_pipeline = PredictPipeline()

        prediction, probability = (
            predict_pipeline.predict(pred_df)
        )

        survival_probability = round(
            probability[0][1] * 100,
            2
        )

        death_probability = round(
            probability[0][0] * 100,
            2
        )

        result = (
            "✅ Survived"
            if int(prediction[0]) == 1
            else "❌ Did Not Survive"
        )

        return {

            "prediction": int(prediction[0]),

            "result": result,

            "survival_probability":
            survival_probability,

            "death_probability":
            death_probability
        }

    except Exception as e:

        return {
            "error": str(e)
        }
#generate for monitor
@app.get("/reports/classification")
def classification_report():

    return FileResponse(
        "artifacts/reports/classification_report.html"
    )
@app.get("/reports/drift")
def drift_report():

    return FileResponse(
        "artifacts/reports/data_drift_report.html"
    )
@app.get("/reports/summary")
def summary_report():

    return FileResponse(
        "artifacts/reports/data_summary_report.html"
    )
@app.get("/dashboard")
def dashboard():

    return FileResponse(
        "templates/Evidently_Dashboard.html"
    )
if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )