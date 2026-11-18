.PHONY: install all test mlflow dashboard metrics clean

install:
	pip install -r requirements.txt

all:
	python -m src.pipeline

test:
	python -m pytest -q

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

dashboard:
	python -m webbrowser reports/monitoring_dashboard.html

metrics:
	python -m src.serve_metrics

clean:
	rm -rf mlflow.db mlartifacts data/processed data/production data/quarantine/*.csv logs/*.jsonl reports/*
