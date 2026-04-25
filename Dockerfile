FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python packages directly (no requirements.txt dependency)
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.5.1 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    pydantic==2.7.4 \
    prometheus-client==0.20.0 \
    setuptools

COPY src/api/ ./src/api/
COPY data/processed/encoders.pkl ./data/processed/

ENV MLFLOW_TRACKING_URI=http://mlflow:5001

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]