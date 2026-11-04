
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

client = mlflow.MlflowClient()

print("\n======================================")
print("DIAGNOSTICO DE MLFLOW")
print("======================================")

print("\nTracking URI:", mlflow.get_tracking_uri())

# 1. Consultar experimentos
print("\n--- EXPERIMENTOS ---")

experiments = client.search_experiments()

for exp in experiments:
    print(f"ID: {exp.experiment_id} | Nombre: {exp.name}")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=100
    )

    print("Ejecuciones:", len(runs))

    for run in runs:
        print(
            run.data.tags.get("mlflow.runName", "Sin nombre"),
            "| F1:",
            run.data.metrics.get("f1_macro", "No registrado")
        )

# 2. Consultar modelos registrados
print("\n--- MODEL REGISTRY ---")

models = client.search_registered_models()

if not models:
    print("No existen modelos registrados.")
else:
    for model in models:
        print("\nModelo:", model.name)

        for version in model.latest_versions:
            print(
                "Version:", version.version,
                "| Run ID:", version.run_id,
                "| Estado:", version.status
            )

print("\n======================================")
print("DIAGNOSTICO FINALIZADO")
print("======================================")
