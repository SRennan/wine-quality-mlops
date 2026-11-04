
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

experiment = client.get_experiment_by_name(
    "Wine Quality Classification"
)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    max_results=100,
    order_by=["attributes.start_time ASC"]
)

print("\n========== DIAGNÓSTICO DE MODELOS ==========")
print("Versión MLflow:", mlflow.__version__)

for run in runs:

    name = run.data.tags.get("mlflow.runName", "Sin nombre")

    if not name.startswith("RF_"):
        continue

    print("\n--------------------------------------------")
    print("Experimento:", name)
    print("Run ID:", run.info.run_id)
    print("Estado:", run.info.status)

    # Artefactos tradicionales
    artifacts = client.list_artifacts(run.info.run_id)

    print("\nArtefactos:")

    if artifacts:
        for artifact in artifacts:
            print(
                artifact.path,
                "| Directorio:", artifact.is_dir
            )
    else:
        print("No hay artefactos tradicionales.")

    # Información de modelos registrados en la ejecución
    print("\nModelos asociados:")

    try:
        if hasattr(client, "search_logged_models"):

            logged_models = client.search_logged_models(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"source_run_id = '{run.info.run_id}'"
            )

            for model in logged_models:
                print("Model ID:", model.model_id)
                print("Artifact location:", model.artifact_location)
                print("Status:", model.status)

        else:
            print("API search_logged_models no disponible.")

    except Exception as error:
        print("Error consultando modelos:", error)

print("\n========== FIN DEL DIAGNÓSTICO ==========")
