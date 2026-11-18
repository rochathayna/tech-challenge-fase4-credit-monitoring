"""Detecção de Data Drift e Concept Drift com Evidently AI + testes próprios (PSI/KS/Qui²).

Para cada lote mensal de produção gera:
  reports/drift/month_XX.html   relatório Evidently (PSI por feature, KS nas numéricas,
                                drift de predição e do target, performance de classificação)
  reports/drift/month_XX.json   resumo estruturado consumido pelo dashboard / MLflow / alertas
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import ClassificationPreset, DataDriftPreset

from .config import (AUC_DROP_ALERT, CATEGORICAL_FEATURES, DRIFT_SHARE_ALERT, FEATURES, KS_PVALUE,
                     NUMERIC_FEATURES, PREDICTION, PREDICTION_PROBA, PSI_WARN, REPORTS_DIR, TARGET)
from .drift_stats import feature_drift_table, psi_numeric
from .train import classification_metrics

DRIFT_DIR = REPORTS_DIR / "drift"


def _definition() -> DataDefinition:
    return DataDefinition(
        numerical_columns=NUMERIC_FEATURES + [PREDICTION_PROBA],
        categorical_columns=CATEGORICAL_FEATURES + [TARGET, PREDICTION],
        classification=[BinaryClassification(target=TARGET, prediction_labels=PREDICTION,
                                             prediction_probas=PREDICTION_PROBA)],
    )


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, html_path, with_performance=True):
    dd = _definition()
    cols = [c for c in reference.columns if c in FEATURES + [TARGET, PREDICTION, PREDICTION_PROBA]]
    ref = Dataset.from_pandas(reference[cols], data_definition=dd)
    cur = Dataset.from_pandas(current[cols], data_definition=dd)
    metrics = [
        DataDriftPreset(columns=FEATURES, method="psi", threshold=PSI_WARN, drift_share=DRIFT_SHARE_ALERT),
        *[ValueDrift(column=c, method="ks", threshold=KS_PVALUE) for c in NUMERIC_FEATURES],
        ValueDrift(column=PREDICTION_PROBA, method="psi", threshold=PSI_WARN),   # drift de predição
        ValueDrift(column=TARGET, method="psi", threshold=PSI_WARN),             # drift do target
    ]
    if with_performance:
        metrics.append(ClassificationPreset())
    snapshot = Report(metrics, include_tests=True).run(cur, ref)
    snapshot.save_html(str(html_path))
    return snapshot.dict()


def _parse_evidently(d: dict) -> dict:
    out = {"psi": {}, "ks_pvalue": {}, "drifted_columns_count": None, "drifted_share": None,
           "tests_total": len(d.get("tests", [])),
           "tests_failed": sum(1 for t in d.get("tests", []) if t.get("status") == "FAIL")}
    for m in d["metrics"]:
        cfg = m.get("config", {})
        if cfg.get("type", "").endswith("ValueDrift"):
            key = "psi" if cfg.get("method") == "psi" else "ks_pvalue"
            out[key][cfg["column"]] = float(m["value"])
        elif cfg.get("type", "").endswith("DriftedColumnsCount"):
            out["drifted_columns_count"] = float(m["value"]["count"])
            out["drifted_share"] = float(m["value"]["share"])
    return out


def analyze_month(month: int, current: pd.DataFrame, reference_scored: pd.DataFrame,
                  baseline_metrics: dict) -> dict:
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    html = DRIFT_DIR / f"month_{month:02d}.html"
    ev = _parse_evidently(evidently_report(reference_scored, current, html))
    custom = feature_drift_table(reference_scored, current)

    perf = classification_metrics(current[TARGET], current[PREDICTION_PROBA])
    approved = current[current[PREDICTION] == 0]
    perf["bad_rate_approved"] = float(approved[TARGET].mean()) if len(approved) else float("nan")
    perf["approval_rate"] = float(len(approved) / len(current))
    pred_psi = psi_numeric(reference_scored[PREDICTION_PROBA], current[PREDICTION_PROBA])

    drifted = custom.loc[custom.drift_detected, "feature"].tolist()
    drift_share = len(drifted) / len(FEATURES)
    auc_drop = baseline_metrics["roc_auc"] - perf["roc_auc"]

    alerts = []
    if drift_share >= DRIFT_SHARE_ALERT:
        alerts.append({"type": "DATASET_DRIFT", "severity": "critical",
                       "message": f"{len(drifted)}/{len(FEATURES)} features com drift ({drift_share:.0%})"})
    for _, r in custom[custom.psi_level == "drift"].iterrows():
        if r.p_value < KS_PVALUE:
            alerts.append({"type": "FEATURE_DRIFT", "severity": "high", "feature": r.feature,
                           "message": f"{r.feature}: PSI={r.psi:.2f} ({r.test} p={r.p_value:.1e})"})
    if pred_psi >= PSI_WARN:
        alerts.append({"type": "PREDICTION_DRIFT", "severity": "high",
                       "message": f"Distribuição do score mudou (PSI={pred_psi:.2f})"})
    if auc_drop >= AUC_DROP_ALERT:
        alerts.append({"type": "CONCEPT_DRIFT", "severity": "critical",
                       "message": f"ROC AUC caiu {auc_drop:.3f} vs. baseline ({perf['roc_auc']:.3f})"})
    gap = perf["default_rate_real"] - perf["default_rate_predicted"]
    if gap >= 0.08:
        alerts.append({"type": "CALIBRATION", "severity": "high",
                       "message": f"Modelo subestima inadimplência em {gap:.1%} (real {perf['default_rate_real']:.1%})"})

    summary = {
        "month": month,
        "rows": int(len(current)),
        "evidently": ev,
        "features": json.loads(custom.to_json(orient="records")),
        "drifted_features": drifted,
        "drift_share": drift_share,
        "prediction_psi": pred_psi,
        "performance": perf,
        "auc_drop": auc_drop,
        "alerts": alerts,
        "status": "critical" if any(a["severity"] == "critical" for a in alerts)
        else ("warning" if alerts else "healthy"),
        "html_report": str(html.relative_to(REPORTS_DIR.parent)),
    }
    (DRIFT_DIR / f"month_{month:02d}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=lambda o: None if o is np.nan else str(o)),
        encoding="utf-8")
    return summary
