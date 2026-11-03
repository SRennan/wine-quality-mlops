import sqlite3

conn = sqlite3.connect("mlflow.db")

cursor = conn.cursor()

cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='table'
""")

print("TABLAS DE MLFLOW")
print("================")

for table in cursor.fetchall():
    print(table[0])

print("\nRUTAS DE EXPERIMENTOS")
print("=====================")

cursor.execute("""
    SELECT experiment_id, name, artifact_location
    FROM experiments
""")

for row in cursor.fetchall():
    print(row)

conn.close()
