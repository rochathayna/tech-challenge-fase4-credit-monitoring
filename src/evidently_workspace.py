"""Evidently UI: workspace local com o histórico mensal de monitoramento.

Cada lote de produção vira um *run* (snapshot) do projeto "Credit Scoring — Produção".
O painel do projeto mostra a evolução de PSI, drift do score, AUC, recall etc.

Para abrir:  evidently ui --workspace evidently_workspace --port 8001
"""
from __future__ import annotations

import shutil

from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import bar_plot_panel, counter_panel, line_plot_panel, text_panel
from evidently.ui.workspace import Workspace

from .config import ROOT

WORKSPACE_DIR = ROOT / "evidently_workspace"
PROJECT_NAME = "Credit Scoring — Produção"

KEY_FEATURES = ["loan_int_rate", "loan_percent_income", "loan_amnt", "loan_grade", "person_income"]


def _psi(column: str, legend: str | None = None) -> PanelMetric:
    return PanelMetric(legend=legend or column, metric="ValueDrift",
                       metric_labels={"column": column, "method": "psi"})


def create_project():
    """Recria o workspace do zero (idempotente) e devolve (workspace, project_id)."""
    shutil.rmtree(WORKSPACE_DIR, ignore_errors=True)
    ws = Workspace.create(str(WORKSPACE_DIR))
    project = ws.create_project(
        PROJECT_NAME,
        description="Monitoramento mensal do modelo XGBoost de risco de crédito: data drift (PSI/KS), "
                    "drift do score e performance com rótulo. Tech Challenge Fase 4.")

    d = project.dashboard
    d.add_panel(text_panel(title="Credit Scoring — monitoramento de produção",
                           description="Um run por lote mensal. Referência = validação do dataset de treino. "
                                       "Limiares: PSI 0,10 atenção · 0,25 drift; AUC −0,05 vs. baseline."),
                tab="Visão geral")
    d.add_panel(counter_panel(title="ROC AUC (último lote)", size="half",
                              values=[PanelMetric(legend="ROC AUC", metric="RocAuc")]), tab="Visão geral")
    d.add_panel(counter_panel(title="Features com drift (último lote)", size="half",
                              values=[PanelMetric(legend="features", metric="DriftedColumnsCount",
                                                  metric_labels={"value_type": "count"})]), tab="Visão geral")
    d.add_panel(line_plot_panel(title="Performance do modelo por lote", size="half",
                                values=[PanelMetric(legend="ROC AUC", metric="RocAuc"),
                                        PanelMetric(legend="Recall", metric="Recall"),
                                        PanelMetric(legend="F1", metric="F1Score")]), tab="Visão geral")
    d.add_panel(bar_plot_panel(title="Fração de features com drift", size="half",
                               values=[PanelMetric(legend="share", metric="DriftedColumnsCount",
                                                   metric_labels={"value_type": "share"})]), tab="Visão geral")
    d.add_panel(line_plot_panel(title="PSI das features-chave", size="full",
                                values=[_psi(c) for c in KEY_FEATURES]), tab="Visão geral")
    d.add_panel(line_plot_panel(title="Drift do score e do target (PSI)", size="full",
                                values=[_psi("prediction_proba", "score do modelo"),
                                        _psi("loan_status", "taxa de inadimplência (target)")]),
                tab="Visão geral")
    d.add_panel(line_plot_panel(title="KS p-valor (escala: < 0,05 = drift)", size="full",
                                values=[PanelMetric(legend=c, metric="ValueDrift",
                                                    metric_labels={"column": c, "method": "ks"})
                                        for c in ["loan_int_rate", "loan_amnt", "person_income", "person_age"]]),
                tab="Testes estatísticos")
    return ws, project.id
