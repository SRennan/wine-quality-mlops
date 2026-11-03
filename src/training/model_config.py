from pathlib import Path

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
MLFLOW_DB = BASE_DIR / "mlflow.db"

TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"

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