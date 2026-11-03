import pandas as pd
from pathlib import Path

# Directorio principal
BASE_DIR = Path(__file__).resolve().parents[2]

# Rutas de los archivos
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def combine_datasets():

    # Cargar vino tinto
    red_wine = pd.read_csv(
        RAW_DIR / "winequality-red.csv",
        sep=";"
    )

    # Cargar vino blanco
    white_wine = pd.read_csv(
        RAW_DIR / "winequality-white.csv",
        sep=";"
    )

    # Identificar el tipo de vino
    red_wine["wine_type"] = "red"
    white_wine["wine_type"] = "white"

    # Combinar ambos datasets
    wine_data = pd.concat(
        [red_wine, white_wine],
        ignore_index=True
    )

    # Guardar dataset combinado
    output_path = PROCESSED_DIR / "wine_quality.csv"

    wine_data.to_csv(
        output_path,
        index=False
    )

    print("COMBINACIÓN COMPLETADA")
    print("----------------------")

    print(f"Vinos tintos: {len(red_wine)}")
    print(f"Vinos blancos: {len(white_wine)}")
    print(f"Total de registros: {len(wine_data)}")

    print(f"\nColumnas: {len(wine_data.columns)}")

    print(f"\nArchivo guardado en: {output_path}")


if __name__ == "__main__":
    combine_datasets()