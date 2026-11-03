import pandas as pd
import json

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================
# CONFIGURACIÓN DE DIRECTORIOS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "wine_quality.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"

EVIDENCE_DIR = (
    BASE_DIR / "evidence" / "data_analysis" / "preprocessing"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# CONFIGURACIÓN
# ==========================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_FEATURES = [
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
    "alcohol"
]


# ==========================================
# FUNCIÓN PRINCIPAL
# ==========================================

def preprocess_data():

    print("\n======================================")
    print("   WINE QUALITY - DATA PREPROCESSING")
    print("======================================")

    # 1. CARGAR DATASET

    df = pd.read_csv(DATA_PATH)

    initial_records = len(df)

    print(f"\nRegistros iniciales: {initial_records}")

    # 2. ELIMINAR DUPLICADOS EXACTOS

    duplicates = int(df.duplicated().sum())

    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Duplicados eliminados: {duplicates}")

    # 3. COMPROBAR VALORES FALTANTES

    missing_values = df.isnull().sum()

    missing_report = missing_values[
        missing_values > 0
    ]

    print("\nValores faltantes:")

    if missing_report.empty:
        print("No existen valores faltantes.")
    else:
        print(missing_report)

    # 4. VALIDAR VARIABLES

    required_columns = (
        NUMERIC_FEATURES + ["quality", "wine_type"]
    )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas: {missing_columns}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Existen valores faltantes. "
            "Se requiere definir su tratamiento antes de continuar."
        )

    # 5. CREAR CATEGORÍAS DE CALIDAD

    df["quality_category"] = pd.cut(
        df["quality"],
        bins=[-float("inf"), 4, 6, float("inf")],
        labels=["low", "medium", "high"]
    )

    print("\nDistribución de categorías:")

    print(
        df["quality_category"].value_counts().sort_index()
    )

    # 6. CODIFICAR TIPO DE VINO

    df = pd.get_dummies(
        df,
        columns=["wine_type"],
        dtype=int
    )

    # 7. DEFINIR VARIABLES DE ENTRADA Y OBJETIVO

    feature_columns = NUMERIC_FEATURES + [
        "wine_type_red",
        "wine_type_white"
    ]

    X = df[feature_columns]

    y = df["quality_category"]

    # 8. DIVIDIR DATOS (80% ENTRENAMIENTO, 20% PRUEBA)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nDIVISIÓN DE DATOS")

    print(f"Entrenamiento: {len(X_train)}")
    print(f"Prueba: {len(X_test)}")

    # 9. ESCALAR VARIABLES NUMÉRICAS

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[NUMERIC_FEATURES] = (
        scaler.fit_transform(
            X_train[NUMERIC_FEATURES]
        )
    )

    X_test_scaled[NUMERIC_FEATURES] = (
        scaler.transform(
            X_test[NUMERIC_FEATURES]
        )
    )

    # 10. CREAR DATASETS FINALES

    train_data = X_train_scaled.copy()

    train_data["quality"] = df.loc[
        X_train.index, "quality"
    ]

    train_data["quality_category"] = y_train

    test_data = X_test_scaled.copy()

    test_data["quality"] = df.loc[
        X_test.index, "quality"
    ]

    test_data["quality_category"] = y_test

    # 11. GUARDAR ARCHIVOS

    train_data.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    test_data.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    df.to_csv(
        OUTPUT_DIR / "wine_quality_clean.csv",
        index=False
    )

    # 12. GENERAR REPORTE

    report = {
        "initial_records": initial_records,
        "removed_duplicates": duplicates,
        "final_records": len(df),
        "missing_values": int(df.isnull().sum().sum()),
        "features": len(feature_columns),
        "training_records": len(train_data),
        "test_records": len(test_data),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "scaling": "StandardScaler",
        "target": "quality_category",
        "categories": {
            "low": "quality <= 4",
            "medium": "quality 5-6",
            "high": "quality >= 7"
        }
    }

    with open(
        EVIDENCE_DIR / "preprocessing_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n======================================")
    print("PREPARACIÓN COMPLETADA")
    print("======================================")

    print(f"Dataset limpio: {OUTPUT_DIR / 'wine_quality_clean.csv'}")
    print(f"Entrenamiento: {OUTPUT_DIR / 'train.csv'}")
    print(f"Prueba: {OUTPUT_DIR / 'test.csv'}")

    print("\nProceso finalizado correctamente.")


if __name__ == "__main__":
    preprocess_data()