.PHONY: install all test mlflow dashboard metrics evidently-ui grafana clean site

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
	python -m src.exporter

clean:
	rm -rf mlflow.db mlartifacts evidently_workspace _site data/processed data/production data/quarantine/*.csv logs/*.jsonl reports/*

evidently-ui:
	evidently ui --workspace evidently_workspace --port 8001

grafana:
	cd monitoring && docker compose up

site:
	python -m src.build_site
