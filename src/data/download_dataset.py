from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

# Directorio principal del proyecto
BASE_DIR = Path(__file__).resolve().parents[2]

# Directorio donde se almacenarán los archivos originales
RAW_DIR = BASE_DIR / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# URL oficial del dataset UCI
URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"


def download_dataset():

    print("Iniciando descarga del dataset Wine Quality...")

    try:
        # Descargar archivo ZIP
        response = urlopen(URL, timeout=60)
        zip_data = BytesIO(response.read())

        # Extraer únicamente los CSV originales
        with ZipFile(zip_data) as zip_file:

            for filename in [
                "winequality-red.csv",
                "winequality-white.csv"
            ]:

                content = zip_file.read(filename)

                output_path = RAW_DIR / filename

                with open(output_path, "wb") as file:
                    file.write(content)

                print(f"Descargado: {filename}")

        print("\nDescarga completada correctamente.")

    except Exception as error:

        print(f"Error durante la descarga: {error}")


if __name__ == "__main__":
    download_dataset()