
import mlflow
import pandas as pd

MODEL_URI = "models:/WineQualityClassifier@champion"

mlflow.set_tracking_uri("http://localhost:5000")

model = mlflow.pyfunc.load_model(MODEL_URI)

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


def predict_wine(data: dict):
    input_data = pd.DataFrame([data], columns=FEATURES)
    prediction = model.predict(input_data)

    return str(prediction[0])
