import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "wine_quality.csv"

OUTPUT_DIR = BASE_DIR / "evidence" / "data_analysis" / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_quality():

    df = pd.read_csv(DATA_PATH)

    # Frecuencia de cada calificación
    quality_counts = (
        df["quality"]
        .value_counts()
        .sort_index()
    )

    # Calcular porcentajes
    quality_percentages = (
        df["quality"]
        .value_counts(normalize=True)
        .sort_index() * 100
    )

    # Crear tabla
    quality_summary = pd.DataFrame({
        "Cantidad": quality_counts,
        "Porcentaje": quality_percentages
    })

    print("\nDISTRIBUCIÓN DE CALIDAD")
    print("-----------------------")

    print(quality_summary)

    # Guardar resultados
    quality_summary.to_csv(
        OUTPUT_DIR.parent / "quality_summary.csv"
    )

    # Gráfico de distribución
    plt.figure(figsize=(9, 5))

    sns.barplot(
        x=quality_counts.index,
        y=quality_counts.values
    )

    plt.title("Distribución de la calidad del vino")
    plt.xlabel("Calidad")
    plt.ylabel("Cantidad de registros")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "quality_distribution_total.png",
        dpi=200
    )

    plt.close()

    # Comparación por tipo de vino
    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=df,
        x="quality",
        hue="wine_type"
    )

    plt.title("Calidad según tipo de vino")
    plt.xlabel("Calidad")
    plt.ylabel("Cantidad")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "quality_by_type.png",
        dpi=200
    )

    plt.close()

    print("\nAnálisis de calidad completado.")


if __name__ == "__main__":
    analyze_quality()