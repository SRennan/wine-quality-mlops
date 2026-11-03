import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Directorios
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "wine_quality.csv"

OUTPUT_DIR = BASE_DIR / "evidence" / "data_analysis" / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_correlation():

    df = pd.read_csv(DATA_PATH)

    # Seleccionar únicamente variables numéricas
    numeric_df = df.select_dtypes(
        include="number"
    )

    # Calcular correlación de Pearson
    correlation = numeric_df.corr(
        method="pearson"
    )

    # Guardar matriz en CSV
    correlation.to_csv(
        OUTPUT_DIR.parent / "correlation_matrix.csv"
    )

    # Crear mapa de calor
    plt.figure(figsize=(12, 9))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True
    )

    plt.title("Matriz de correlación - Wine Quality")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlation_matrix.png",
        dpi=200
    )

    plt.close()

    # Mostrar correlaciones con quality
    quality_correlation = (
        correlation["quality"]
        .drop("quality")
        .sort_values(ascending=False)
    )

    print("\nCORRELACIÓN CON QUALITY")
    print("-----------------------")

    print(quality_correlation)

    quality_correlation.to_csv(
        OUTPUT_DIR.parent / "quality_correlations.csv"
    )

    print("\nMatriz de correlación generada correctamente.")


if __name__ == "__main__":
    generate_correlation()