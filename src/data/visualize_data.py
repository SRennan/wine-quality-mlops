import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Directorios
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "wine_quality.csv"

OUTPUT_DIR = BASE_DIR / "evidence" / "data_analysis" / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuración visual
sns.set_theme(style="whitegrid")


def generate_visualizations():

    df = pd.read_csv(DATA_PATH)

    # Seleccionar variables numéricas
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    # Excluir quality de las variables fisicoquímicas
    features = [
        column for column in numeric_columns
        if column != "quality"
    ]

    # 1. Histogramas
    print("Generando histogramas...")

    for column in features:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=column,
            kde=True
        )

        plt.title(f"Distribución de {column}")
        plt.xlabel(column)
        plt.ylabel("Frecuencia")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"histogram_{column}.png",
            dpi=150
        )

        plt.close()

    # 2. Diagramas de caja
    print("Generando diagramas de caja...")

    for column in features:

        plt.figure(figsize=(8, 4))

        sns.boxplot(
            data=df,
            x=column
        )

        plt.title(f"Boxplot de {column}")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"boxplot_{column}.png",
            dpi=150
        )

        plt.close()

    # 3. Distribución de calidad
    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="quality",
        hue="wine_type"
    )

    plt.title("Distribución de calidad por tipo de vino")
    plt.xlabel("Calidad")
    plt.ylabel("Cantidad de registros")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "quality_distribution.png",
        dpi=150
    )

    plt.close()

    print("\nVisualizaciones generadas correctamente.")
    print(f"Ubicación: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_visualizations()