from pathlib import Path
import os

# Directorio principal
BASE_DIR = Path(__file__).resolve().parents[2]

# Dataset
DATA_PATH = (
    BASE_DIR /
    "data" /
    "processed" /
    "wine_quality_clean.csv"
)


# Base de datos SQLite para MLflow
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"
)

TRACKING_URI = MLFLOW_TRACKING_URI

# Nombre del experimento
EXPERIMENT_NAME = "Wine Quality Classification"

# Configuración de entrenamiento
RANDOM_STATE = 42

TEST_SIZE = 0.20

# Variables de entrada
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "wine_type_red",
    "wine_type_white"
]

# Variable objetivo
TARGET = "quality_category"