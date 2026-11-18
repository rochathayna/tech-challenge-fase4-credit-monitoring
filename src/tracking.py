"""Configuração do MLflow (backend SQLite local; troque por um servidor remoto em produção)."""
import mlflow

from .config import ROOT

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"


def setup_mlflow(experiment: str) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(ROOT / "mlartifacts" / experiment).as_uri())
    exp = mlflow.set_experiment(experiment)
    # MLflow 3 abre experimentos como "GenAI" por padrão; marca como treino de modelo clássico
    mlflow.MlflowClient().set_experiment_tag(exp.experiment_id, "mlflow.experimentKind", "custom_model_development")
