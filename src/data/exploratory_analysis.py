import pandas as pd
from pathlib import Path

# Directorios
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "wine_quality.csv"

REPORT_DIR = BASE_DIR / "evidence" / "data_analysis"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def exploratory_analysis():

    # Cargar dataset
    df = pd.read_csv(DATA_PATH)

    print("\n===================================")
    print("   WINE QUALITY - DATA ANALYSIS")
    print("===================================\n")

    # 1. Información general
    print("1. INFORMACIÓN GENERAL")

    print(f"Registros: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")

    print("\nTipos de variables:")
    print(df.dtypes)

    # 2. Valores faltantes
    print("\n2. VALORES FALTANTES")

    missing = df.isnull().sum()

    print(missing)

    missing.to_csv(
        REPORT_DIR / "missing_values.csv"
    )

    # 3. Registros duplicados
    print("\n3. REGISTROS DUPLICADOS")

    duplicates = df.duplicated().sum()

    print(f"Duplicados encontrados: {duplicates}")

    # 4. Estadísticas descriptivas
    print("\n4. ESTADÍSTICAS DESCRIPTIVAS")

    statistics = df.describe().T

    print(statistics)

    statistics.to_csv(
        REPORT_DIR / "descriptive_statistics.csv"
    )

    # 5. Distribución del tipo de vino
    print("\n5. DISTRIBUCIÓN DEL TIPO DE VINO")

    print(df["wine_type"].value_counts())

    df["wine_type"].value_counts().to_csv(
        REPORT_DIR / "wine_type_distribution.csv"
    )

    # 6. Distribución de calidad
    print("\n6. DISTRIBUCIÓN DE CALIDAD")

    print(df["quality"].value_counts().sort_index())

    df["quality"].value_counts().sort_index().to_csv(
        REPORT_DIR / "quality_distribution.csv"
    )

    # 7. Guardar información general
    with open(
        REPORT_DIR / "dataset_information.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("WINE QUALITY DATASET\n")
        file.write("====================\n\n")

        file.write(f"Registros: {df.shape[0]}\n")
        file.write(f"Columnas: {df.shape[1]}\n")
        file.write(f"Duplicados: {duplicates}\n")
        file.write(f"Valores faltantes: {df.isnull().sum().sum()}\n")

    print("\nAnálisis estadístico completado.")
    print(f"Reportes guardados en: {REPORT_DIR}")


if __name__ == "__main__":
    exploratory_analysis()