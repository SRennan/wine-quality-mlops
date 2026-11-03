from fastapi import FastAPI

from api.schemas import (
    WineInput,
    PredictionResponse
)

from api.predictor import (
    predict,
    get_model_info,
    get_pod_name
)


# ==========================================
# CREAR API
# ==========================================

app = FastAPI(
    title="Wine Quality Prediction API",
    description="API para predecir la calidad del vino utilizando Machine Learning y MLflow.",
    version="1.0.0"
)


# ==========================================
# ENDPOINT PRINCIPAL
# ==========================================

@app.get("/")
def root():

    return {
        "message": "Wine Quality Prediction API",
        "status": "online"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "pod_name": get_pod_name()
    }


# ==========================================
# INFORMACIÓN DEL MODELO
# ==========================================

@app.get("/model-info")
def model_info():

    return get_model_info()


# ==========================================
# ENDPOINT DE PREDICCIÓN
# ==========================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)

def predict_wine(data: WineInput):

    prediction = predict(data)

    return {
        "wine_type": data.wine_type,
        "prediction": prediction,
        "model_name": "WineQualityClassifier",
        "model_version": get_model_info()["model_version"],
        "model_alias": "production",
        "pod_name": get_pod_name()
    }
