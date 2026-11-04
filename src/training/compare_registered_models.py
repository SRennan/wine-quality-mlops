
import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

MODEL_NAME = "WineQualityClassifier"

print("\n======================================")
print("COMPARACIÓN DE MODELOS REGISTRADOS")
print("======================================")

versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

results = []

for version in versions:

    run = client.get_run(version.run_id)

    metrics = run.data.metrics

    results.append({
        "version": int(version.version),
        "experiment": version.tags.get(
            "training_experiment", "Sin identificar"
        ),
        "accuracy": metrics.get("accuracy", 0),
        "precision_macro": metrics.get("precision_macro", 0),
        "recall_macro": metrics.get("recall_macro", 0),
        "f1_macro": metrics.get("f1_macro", 0),
        "status": version.status
    })

df = pd.DataFrame(results)

df = df.sort_values(
    by="f1_macro",
    ascending=False
)

print(df.to_string(index=False))

# Guardar comparación
df.to_csv(
    "evidence/mlflow/registered_models_comparison.csv",
    index=False
)

print("\nMejor modelo:")
print(df.iloc[0].to_string())

print("\nTotal de modelos:", len(df))

