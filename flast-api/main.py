from typing import Literal, List

import numpy as np
import pandas as pd

import sklearn
import xgboost
import lightgbm
import catboost

from fastapi import FastAPI, Path, HTTPException
from pydantic import BaseModel, Field

from features import compute_features
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


class RawInput(BaseModel):
    accelerometer: Literal["DE", "FE"]
    raw: List[List[float]] = Field(..., description="Lista de listas de señales crudas. Cada lista debe tener 2048 elementos.")


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


@app.post("/predictraw/{model}")
def predict_raw(model: str = Path(..., description="Nombre del modelo (run name en MLflow)"),
                raw_input: RawInput = ...):
    try:
        loaded_model = load_model_from_mlflow(model)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Validación: asegurar que cada lista interna tenga 2048 elementos
    for i, signal in enumerate(raw_input.raw):
        if len(signal) != 2048:
            raise HTTPException(
                status_code=400,
                detail=f"La señal en la posición {i} no tiene 2048 elementos."
            )

    # Computar características desde señales crudas
    try:
        input_df = compute_features(np.array(raw_input.raw))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en compute_features: {str(e)}")

    input_df["accelerometer"] = raw_input.accelerometer
    input_df = input_df.fillna(0)

    print(input_df.to_dict(orient='records'))

    # Predicción
    try:
        prediction = loaded_model.predict(input_df.to_dict(orient='records'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {str(e)}")

    return {"prediction": prediction.tolist()}
