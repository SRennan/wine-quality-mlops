import sqlite3

conn = sqlite3.connect("mlflow.db")

cursor = conn.cursor()

cursor.execute("""
    SELECT
        name,
        version,
        run_id,
        source,
        storage_location
    FROM model_versions
""")

print("\nMODELOS REGISTRADOS")
print("===================")

for row in cursor.fetchall():
    print("\nNombre:", row[0])
    print("Versión:", row[1])
    print("Run ID:", row[2])
    print("Source:", row[3])
    print("Storage:", row[4])

conn.close()