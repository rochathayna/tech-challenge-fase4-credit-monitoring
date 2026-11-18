"""Configuração do MLflow (backend SQLite local; troque por um servidor remoto em produção)."""
import mlflow

from .config import ROOT

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"


def setup_mlflow(experiment: str) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(ROOT / "mlartifacts" / experiment).as_uri())
    mlflow.set_experiment(experiment)
