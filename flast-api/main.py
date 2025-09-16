import sklearn
import xgboost
import lightgbm
import catboost

from fastapi import FastAPI, Path, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd

from model_utils import load_model_from_mlflow

app = FastAPI(title="MLflow Prediction API")



class FeatureInput(BaseModel):
    maximum: float
    mean: float
    std: float
    rms: float
    skewness: float
    kurtosis: float
    crest_factor: float
    form_factor: float
    accelerometer: Literal["DE", "FE"]
    thd: float
    f0: int



@app.post("/predictfeatures/{model}")
def predict(model: str = Path(..., description="Nombre del modelo (run name en MLflow)"),
            features: FeatureInput = ...):
    try:
        loaded_model = load_model_from_mlflow(model)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Convertimos los datos a DataFrame
    input_df = pd.DataFrame([features.model_dump()])

    # Realizamos la predicción
    try:
        prediction = loaded_model.predict(input_df.to_dict(orient='records'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {str(e)}")

    return {"prediction": prediction.tolist()}


