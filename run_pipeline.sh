#!/bin/bash
set -e

echo "=== Activating environment ==="
source venv/bin/activate

echo "=== Starting MLflow server ==="
mlflow server \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 --port 5002 &
sleep 5

echo "=== Running pipeline ==="
python src/pipeline/01_ingest.py
python src/pipeline/02_validate.py
python src/pipeline/03_preprocess.py

for strategy in undersample class_weight; do
  echo "--- Balancing strategy: $strategy ---"
  python src/pipeline/04_features.py $strategy
done

python src/pipeline/05_train.py undersample
python src/pipeline/06_deploy.py

echo "=== Simulating drift ==="
python src/pipeline/07_drift_sim.py

echo "=== Checking drift ==="
python src/monitoring/drift_monitor.py

echo "=== Starting API ==="
python src/api/app.py &

echo ""
echo "All done!"
echo "MLflow UI:  http://localhost:5000"
echo "API:        http://localhost:8000"
echo "API docs:   http://localhost:8000/docs"
echo "Prometheus: http://localhost:9090"
echo "Grafana:    http://localhost:3000  (admin/admin)"
