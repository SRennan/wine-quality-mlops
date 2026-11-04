import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

from mlflow.models import infer_signature

from src.training.model_config import (
    BASE_DIR,
    DATA_PATH,
    TRACKING_URI,
    EXPERIMENT_NAME,
    RANDOM_STATE,
    TEST_SIZE,
    FEATURES,
    TARGET
)


# ==========================================
# CONFIGURACIÓN DE MLFLOW
# ==========================================

# Configurar almacenamiento SQLite
mlflow.set_tracking_uri(TRACKING_URI)

# Crear o recuperar experimento
mlflow.set_experiment(EXPERIMENT_NAME)

# ==========================================
# CONFIGURACIÓN DE EXPERIMENTOS
# ==========================================

EXPERIMENTS = [
    {
        "name": "RF_01",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2
    },
    {
        "name": "RF_02",
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 2
    },
    {
        "name": "RF_03",
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 2
    },
    {
        "name": "RF_04",
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5
    },
    {
        "name": "RF_05",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 5
    },
        {
        "name": "RF_06",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "class_weight": "balanced"
    },
    {
        "name": "RF_07",
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 2,
        "class_weight": "balanced"
    },
    {
        "name": "RF_08",
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 2,
        "class_weight": "balanced"
    },
    {
        "name": "RF_09",
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "class_weight": "balanced"
    },
    {
        "name": "RF_10",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 5,
        "class_weight": "balanced"
    }
]


# ==========================================
# CARGAR Y PREPARAR DATOS
# ==========================================

def load_data():

    print("\nCargando dataset...")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].astype("float64")

    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Total registros: {len(df)}")
    print(f"Entrenamiento: {len(X_train)}")
    print(f"Prueba: {len(X_test)}")

    return X_train, X_test, y_train, y_test


# ==========================================
# ENTRENAR Y REGISTRAR EXPERIMENTO
# ==========================================

def run_experiment(config, X_train, X_test, y_train, y_test):

    print("\n====================================")
    print(f"EXPERIMENTO: {config['name']}")
    print("====================================")

    with mlflow.start_run(run_name=config["name"]):

        # 1. Registrar hiperparámetros

        mlflow.log_params({
            "algorithm": "RandomForestClassifier",
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "min_samples_split": config["min_samples_split"],
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "class_weight": config.get("class_weight", None)
        })

        # 2. Crear pipeline

        model = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=config["n_estimators"],
                    max_depth=config["max_depth"],
                    min_samples_split=config["min_samples_split"],
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight=config.get("class_weight", None)
                )
            )
        ])

        # 3. Entrenar

        model.fit(X_train, y_train)

        # 4. Realizar predicciones

        predictions = model.predict(X_test)

        # 5. Calcular métricas

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision_macro": precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            ),
            "recall_macro": recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            ),
            "f1_macro": f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
        }

        # 6. Registrar métricas

        mlflow.log_metrics(metrics)

        # 7. Registrar etiquetas

        mlflow.set_tags({
            "problem_type": "multiclass_classification",
            "dataset": "Wine Quality",
            "algorithm": "Random Forest",
            "target": TARGET
        })

        # 8. Generar firma del modelo

        signature = infer_signature(
            X_train,
            model.predict(X_train)
        )

        # 9. Registrar el modelo como artefacto

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_train.head(5),

            # Autorizar el tipo interno utilizado por Random Forest
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # 10. Registrar reporte de clasificación

        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        report_path = (
            BASE_DIR /
            "evidence" /
            "mlflow" /
            f"{config['name']}_classification_report.csv"
        )

        report_path.parent.mkdir(parents=True, exist_ok=True)

        report_df.to_csv(report_path)

        mlflow.log_artifact(str(report_path))


        # 11. Generar matriz de confusión

        cm = confusion_matrix(y_test, predictions)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        disp.plot(cmap="Blues")

        plt.title(f"Matriz de confusión - {config['name']}")

        plt.tight_layout()

        # Guardar imagen
        matrix_path = (
            BASE_DIR /
            "evidence" /
            "mlflow" /
            f"{config['name']}_confusion_matrix.png"
        )

        plt.savefig(
            matrix_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # Registrar imagen en MLflow
        mlflow.log_artifact(str(matrix_path))

        print(f"Matriz de confusión guardada: {matrix_path}")


        # 12. Mostrar resultados

        print("\nRESULTADOS:")

        for metric, value in metrics.items():

            print(f"{metric}: {value:.4f}")

        print("\nExperimento registrado correctamente.")

        return metrics


# ==========================================
# EJECUTAR TODOS LOS EXPERIMENTOS
# ==========================================

def main():

    X_train, X_test, y_train, y_test = load_data()

    results = []

    # Segundo grupo: modelos con balanceo
    group_2 = EXPERIMENTS[5:10]

    print("\n====================================")
    print("GRUPO 2: MODELOS CON BALANCEO")
    print("====================================")

    for config in group_2:

        metrics = run_experiment(
            config,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append({
            "experiment": config["name"],
            "class_weight": "balanced",
            **metrics
        })

    results_df = pd.DataFrame(results)

    output_path = (
        BASE_DIR /
        "evidence" /
        "mlflow" /
        "experiment_comparison_group_2.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path, index=False)

    print("\n====================================")
    print("RESULTADOS GRUPO 2")
    print("====================================")

    print(results_df.to_string(index=False))

    print("\nGrupo 2 finalizado.")


if __name__ == "__main__":
    main()