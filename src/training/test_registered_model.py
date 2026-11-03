import mlflow
import pandas as pd

from src.training.model_config import TRACKING_URI

# ==========================================
# CONFIGURACIÓN
# ==========================================

mlflow.set_tracking_uri(TRACKING_URI)

MODEL_URI = "models:/WineQualityClassifier@production"

# ==========================================
# CARGAR MODELO REGISTRADO
# ==========================================

print("\nCargando modelo desde MLflow Registry...")

model = mlflow.sklearn.load_model(MODEL_URI)

print("Modelo cargado correctamente.")

# ==========================================
# DATOS DE PRUEBA
# ==========================================

sample = pd.DataFrame([{
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "wine_type_red": 1,
    "wine_type_white": 0
}])

# ==========================================
# REALIZAR PREDICCIÓN
# ==========================================

prediction = model.predict(sample)



print("\nRESULTADO DE PREDICCIÓN")
print("=======================")

print("Categoría:", str(prediction[0]).capitalize())

print("\nPrueba finalizada correctamente.")