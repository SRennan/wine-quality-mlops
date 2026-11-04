FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /mlflow/db /mlflow/mlruns

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/db/mlflow.db", \
     "--default-artifact-root", "file:///mlflow/mlruns", \
     "--workers", "1", \
     "--allowed-hosts", "*"]
