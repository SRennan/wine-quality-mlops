import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CARGAR DATASET
# ==========================================

df = pd.read_csv(
    "data/processed/wine_quality_clean.csv"
)


# ==========================================
# DISTRIBUCIÓN DE CATEGORÍAS
# ==========================================

print("\nDISTRIBUCIÓN DE CATEGORÍAS")
print("==========================")

counts = df["quality_category"].value_counts()

print(counts)


# ==========================================
# PORCENTAJES
# ==========================================

print("\nPORCENTAJES")
print("===========")

percentages = (
    df["quality_category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(percentages)


# ==========================================
# TOTAL DE REGISTROS
# ==========================================

print("\nTOTAL DE REGISTROS")
print("==================")

print(len(df))


# ==========================================
# GRÁFICO DE DISTRIBUCIÓN
# ==========================================

plt.figure(figsize=(9, 5))

counts.plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Distribución de categorías de calidad del vino")

plt.xlabel("Categoría de calidad")

plt.ylabel("Cantidad de registros")

plt.xticks(rotation=0)

plt.tight_layout()


# ==========================================
# GUARDAR GRÁFICO
# ==========================================

output_path = "evidence/mlflow/class_distribution.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nGráfico guardado en: {output_path}")