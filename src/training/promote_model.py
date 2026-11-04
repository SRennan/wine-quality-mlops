
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()

MODEL_NAME = "WineQualityClassifier"
BEST_VERSION = "7"

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=BEST_VERSION
)

model = client.get_model_version_by_alias(
    name=MODEL_NAME,
    alias="champion"
)

print("MODELO SELECCIONADO PARA PRODUCCIÓN")
print("==================================")
print("Nombre:", model.name)
print("Versión:", model.version)
print("Estado:", model.status)
print("Alias: champion")
print("URI: models:/WineQualityClassifier@champion")
