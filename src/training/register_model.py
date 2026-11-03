import mlflow
from mlflow import MlflowClient

from src.training.model_config import (
    TRACKING_URI,
    EXPERIMENT_NAME
)

# ==========================================
# CONFIGURACIÓN
# ==========================================

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient(
    tracking_uri=TRACKING_URI
)

MODEL_NAME = "WineQualityClassifier"

# ==========================================
# BUSCAR EXPERIMENTO
# ==========================================

experiment = client.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:
    raise ValueError("Experimento no encontrado.")

# ==========================================
# BUSCAR RF_07
# ==========================================

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.run_name = 'RF_07'",
    order_by=["attributes.start_time DESC"]
)

if not runs:
    raise ValueError("No se encontró RF_07.")

# Seleccionar la ejecución más reciente
run = runs[0]

run_id = run.info.run_id

print("\nMODELO SELECCIONADO")
print("===================")

print("Run ID:", run_id)

print(
    "F1 Macro:",
    run.data.metrics["f1_macro"]
)

# ==========================================
# REGISTRAR MODELO
# ==========================================

model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print("\nMODELO REGISTRADO")
print("=================")

print("Nombre:", MODEL_NAME)
print("Versión:", result.version)

# ==========================================
# ASIGNAR ALIAS
# ==========================================

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="production",
    version=result.version
)

print("\nALIAS ASIGNADO")
print("=================")

print("Modelo:", MODEL_NAME)
print("Alias: production")
print("Versión:", result.version)