import sqlite3

conn = sqlite3.connect("mlflow.db")
cursor = conn.cursor()

# =====================================
# 1. RUTAS DE EXPERIMENTOS
# =====================================

print("\nEXPERIMENTOS")
print("=" * 60)

cursor.execute("""
    SELECT experiment_id, name, artifact_location
    FROM experiments
""")

for row in cursor.fetchall():
    print(row)


# =====================================
# 2. RUTAS DE RUNS
# =====================================

print("\nRUNS")
print("=" * 60)

cursor.execute("""
    SELECT run_uuid, artifact_uri
    FROM runs
    LIMIT 5
""")

for row in cursor.fetchall():
    print(row)


# =====================================
# 3. MODELOS LOGUEADOS
# =====================================

print("\nLOGGED MODELS")
print("=" * 60)

cursor.execute("""
    SELECT model_id, artifact_location
    FROM logged_models
    LIMIT 5
""")

for row in cursor.fetchall():
    print(row)


# =====================================
# 4. VERSIONES DEL REGISTRY
# =====================================

print("\nMODEL VERSIONS")
print("=" * 60)

cursor.execute("""
    SELECT
        name,
        version,
        source,
        storage_location
    FROM model_versions
""")

for row in cursor.fetchall():
    print(row)


conn.close()
