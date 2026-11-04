
import mlflow
from mlflow.tracking import MlflowClient

# ==========================================
# CONFIGURACIÓN
# ==========================================

TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Wine Quality Classification"
MODEL_NAME = "WineQualityClassifier"

SELECTED_EXPERIMENTS = [
    "RF_01", "RF_02", "RF_03", "RF_04", "RF_05",
    "RF_06", "RF_07", "RF_08", "RF_09", "RF_10"
]

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# ==========================================
# BUSCAR EXPERIMENTOS Y MODELOS
# ==========================================

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    max_results=100,
    order_by=["attributes.start_time ASC"]
)

selected = {}

for run in runs:

    name = run.data.tags.get("mlflow.runName")

    if (
        name in SELECTED_EXPERIMENTS
        and run.info.status == "FINISHED"
        and name not in selected
    ):
        selected[name] = run

# ==========================================
# RECUPERAR LOGGED MODELS
# ==========================================

logged_models = client.search_logged_models(
    experiment_ids=[experiment.experiment_id]
)

model_by_run = {}

for model in logged_models:

    run_id = model.source_run_id

    if model.status == "READY":
        model_by_run[run_id] = model

# ==========================================
# CREAR MODEL REGISTRY
# ==========================================

try:
    client.create_registered_model(MODEL_NAME)
    print("Modelo registrado creado.")

except Exception:
    print("El modelo registrado ya existe.")

# ==========================================
# REGISTRAR DIEZ VERSIONES
# ==========================================

print("\n======================================")
print("REGISTRO DE MODELOS")
print("======================================")

for name in SELECTED_EXPERIMENTS:

    run = selected.get(name)

    if run is None:
        raise RuntimeError(f"No se encontró {name}")

    logged_model = model_by_run.get(run.info.run_id)

    if logged_model is None:
        raise RuntimeError(
            f"No se encontró el Logged Model de {name}"
        )

    # Verificar si ya está registrado ese modelo
    existing = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    existing_model_ids = {
        version.tags.get("mlflow.source.modelId")
        for version in existing
    }

    if logged_model.model_id in existing_model_ids:
        print(f"{name}: Ya registrado.")
        continue

    version = mlflow.register_model(
        model_uri=f"models:/{logged_model.model_id}",
        name=MODEL_NAME,
        await_registration_for=120
    )

    client.set_model_version_tag(
        MODEL_NAME,
        version.version,
        "training_experiment",
        name
    )

    print(
        f"{name} -> Versión {version.version} "
        f"| Estado: {version.status}"
    )

# ==========================================
# COMPROBAR RESULTADO
# ==========================================

print("\n======================================")
print("MODEL REGISTRY FINAL")
print("======================================")

versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

versions = sorted(
    versions,
    key=lambda v: int(v.version)
)

for version in versions:

    print(
        f"Versión {version.version} | "
        f"Experimento: "
        f"{version.tags.get('training_experiment')} | "
        f"Estado: {version.status}"
    )

print("\nTotal de versiones:", len(versions))
