FROM python:3.12-slim

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependencias
COPY requirements.txt .

# Instalar bibliotecas Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar proyecto
COPY api/ ./api/
COPY src/ ./src/

# Puerto de FastAPI
EXPOSE 8000

# Ejecutar API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]