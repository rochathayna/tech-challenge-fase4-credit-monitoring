"""Orquestrador ponta a ponta:  python -m src.pipeline

  1. ingest_and_prepare  pseudonimização + limpeza + split Referência/Holdout
  2. data_contracts      contrato Pandera: bruto reprovado, referência aprovada, lote corrompido bloqueado
  3. training            baseline XGBoost (MLflow)
  4. drift_simulation    6 lotes mensais com data drift + concept drift
  5. production_monitoring  por lote: contrato -> scoring (latência) -> Evidently/PSI/KS -> alertas -> MLflow
  6. causal_analysis     intervenções por mecanismo
  7. observability       Prometheus + dashboard HTML consolidado
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from .causal import run_causal_analysis  # noqa: E402
from .config import FEATURES, MODEL_PATH, PRODUCTION_MONTHS, REPORTS_DIR, TARGET  # noqa: E402
from .data_prep import build_datasets  # noqa: E402
from .detect_drift import analyze_month  # noqa: E402
from .evidently_workspace import create_project  # noqa: E402
from .logging_utils import RUN_ID, log_event, read_logs, stage_timer  # noqa: E402
from .monitoring import build_dashboard, export_prometheus, log_month_to_mlflow  # noqa: E402
from .simulate_drift import build_production_batches  # noqa: E402
from .train import REFERENCE_SCORED_PATH, score, train  # noqa: E402
from .validate import ingest, make_corrupted_batch, run_contract_demo  # noqa: E402

CORRUPTED_MONTH = 4   # mês em que o parceiro envia um arquivo quebrado antes do reenvio


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_event("pipeline", "pipeline_started", "info", run_id=RUN_ID)

    with stage_timer("ingest_and_prepare"):
        d = build_datasets()
        log_event("ingest_and_prepare", "datasets_built", "success", raw_rows=len(d["raw"]),
                  reference_rows=len(d["reference"]), holdout_rows=len(d["holdout"]),
                  removed_by_cleaning=d["removed_by_cleaning"])

    with stage_timer("data_contracts"):
        run_contract_demo(d["reference"], d["pseudo"])

    with stage_timer("training"):
        baseline = train()["metrics"]

    with stage_timer("drift_simulation"):
        batches = build_production_batches()

    model = joblib.load(MODEL_PATH)
    reference_scored = pd.read_csv(REFERENCE_SCORED_PATH)
    summaries = []
    workspace, project_id = create_project()
    with stage_timer("production_monitoring"):
        for month, batch in batches.items():
            name = f"production_month_{month:02d}"
            features_only = batch.drop(columns=[TARGET])   # no momento do scoring o rótulo ainda não existe

            if month == CORRUPTED_MONTH:
                bad = make_corrupted_batch(batch).drop(columns=[TARGET])
                if ingest(bad, f"{name}_envio_corrompido", require_target=False) is None:
                    log_event("production_monitoring", "partner_resend_requested", "warning", batch=name)

            accepted = ingest(features_only, name, require_target=False)
            if accepted is None:
                continue

            t0 = time.perf_counter()
            scored = score(model, accepted)
            latency_ms = (time.perf_counter() - t0) * 1000
            log_event("scoring", "batch_scored", "success", batch=name, rows=len(scored),
                      latency_ms=round(latency_ms, 1), throughput_rows_s=round(len(scored) / (latency_ms / 1000)),
                      predicted_default_rate=round(float(scored["prediction"].mean()), 4))

            # rótulos chegam depois (performance window de ~90 dias) e são unidos por client_id
            labeled = scored.merge(batch[["client_id", TARGET]], on="client_id")
            summary = analyze_month(month, labeled, reference_scored, baseline, workspace, project_id)
            summaries.append(summary)
            for a in summary["alerts"]:
                log_event("drift_detection", "alert", "alert", batch=name, alert_type=a["type"],
                          severity=a["severity"], message=a["message"])
            log_event("drift_detection", "batch_analyzed", summary["status"], batch=name,
                      drift_share=round(summary["drift_share"], 3), roc_auc=round(summary["performance"]["roc_auc"], 4),
                      alerts=len(summary["alerts"]), drifted=summary["drifted_features"])
            log_month_to_mlflow(summary, latency_ms)

    with stage_timer("causal_analysis"):
        causal = run_causal_analysis(reference_scored)

    with stage_timer("observability"):
        prom = export_prometheus(summaries, read_logs())
        dash = build_dashboard(summaries, baseline, causal)
        log_event("observability", "artifacts_written", "success", prometheus=prom, dashboard=dash)

    overview = {
        "run_id": RUN_ID, "baseline": baseline,
        "months": [{"month": s["month"], "status": s["status"], "roc_auc": s["performance"]["roc_auc"],
                    "bad_rate_approved": s["performance"]["bad_rate_approved"],
                    "drifted_features": s["drifted_features"], "alerts": [a["type"] for a in s["alerts"]]}
                   for s in summaries],
        "features": FEATURES, "production_months": PRODUCTION_MONTHS,
    }
    (REPORTS_DIR / "run_summary.json").write_text(json.dumps(overview, indent=2, ensure_ascii=False), encoding="utf-8")
    log_event("pipeline", "pipeline_finished", "success", run_id=RUN_ID)


if __name__ == "__main__":
    main()
