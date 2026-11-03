from pydantic import BaseModel, Field
from typing import Literal


# ==========================================
# DATOS DE ENTRADA
# ==========================================

class WineInput(BaseModel):

    fixed_acidity: float = Field(alias="fixed acidity")
    volatile_acidity: float = Field(alias="volatile acidity")
    citric_acid: float = Field(alias="citric acid")
    residual_sugar: float = Field(alias="residual sugar")
    chlorides: float
    free_sulfur_dioxide: float = Field(alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(alias="total sulfur dioxide")
    density: float
    pH: float
    sulphates: float
    alcohol: float

    wine_type: Literal["red", "white"]

    model_config = {
        "populate_by_name": True
    }


# ==========================================
# DATOS DE RESPUESTA
# ==========================================

class PredictionResponse(BaseModel):

    wine_type: str
    prediction: str
    model_name: str
    model_version: str
    model_alias: str
    pod_name: str