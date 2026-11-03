import mlflow
import mlflow.sklearn

import pandas as pd
import os
import socket

from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db"
)


# ==========================================
# CONFIGURACIÓN
# ==========================================

MODEL_NAME = "WineQualityClassifier"
MODEL_ALIAS = "production"

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()


# ==========================================
# CARGAR MODELO
# ==========================================

print("\nConectando con MLflow Registry...")

model = mlflow.sklearn.load_model(MODEL_URI)

model_version = client.get_model_version_by_alias(
    name=MODEL_NAME,
    alias=MODEL_ALIAS
)

MODEL_VERSION = str(model_version.version)

# Recuperar las variables originales del entrenamiento
EXPECTED_FEATURES = list(
    model.feature_names_in_
)

print("Modelo cargado correctamente.")
print("Modelo:", MODEL_NAME)
print("Versión:", MODEL_VERSION)


# ==========================================
# IDENTIFICAR POD
# ==========================================

def get_pod_name():

    return os.getenv(
        "HOSTNAME",
        socket.gethostname()
    )


# ==========================================
# PREPARAR DATOS
# ==========================================

def prepare_input(data):

    # Convertir entrada en diccionario
    data = data.model_dump(by_alias=True)

    wine_type = data.pop("wine_type")

    # Crear variables One-Hot Encoding
    data["wine_type_red"] = 1 if wine_type == "red" else 0

    data["wine_type_white"] = 1 if wine_type == "white" else 0

    # Construir DataFrame con el orden exacto
    # de las variables utilizadas durante el entrenamiento

    input_df = pd.DataFrame(
        [[data[feature] for feature in EXPECTED_FEATURES]],
        columns=EXPECTED_FEATURES
    )

    return input_df


# ==========================================
# REALIZAR PREDICCIÓN
# ==========================================

def predict(data):

    input_df = prepare_input(data)

    prediction = model.predict(input_df)

    return str(prediction[0])


# ==========================================
# INFORMACIÓN DEL MODELO
# ==========================================

def get_model_info():

    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_alias": MODEL_ALIAS
    }