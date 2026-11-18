"""Exporter Prometheus das métricas de saúde do pipeline e do modelo.

Lê as evidências geradas por `python -m src.pipeline` (reports/drift/*.json e logs/pipeline.jsonl)
e expõe em http://localhost:8000/metrics.

Modos:
  python -m src.exporter                 expõe o estado do último lote
  python -m src.exporter --replay 15     "reproduz" a produção: avança um mês a cada 15 s (e recomeça),
                                         para o Grafana mostrar a degradação acontecendo ao vivo
"""
from __future__ import annotations

import argparse
import json
import time

from prometheus_client import Counter, Gauge, start_http_server

from .config import PRODUCTION_MONTHS, REPORTS_DIR, ROOT

LOG_FILE = ROOT / "logs" / "pipeline.jsonl"
STATUS_CODE = {"healthy": 0, "warning": 1, "critical": 2}

g_month = Gauge("credit_current_month", "Mês do lote de produção exposto")
g_status = Gauge("credit_model_status", "Status do modelo: 0 saudável, 1 atenção, 2 crítico")
g_auc = Gauge("credit_model_roc_auc", "ROC AUC do lote (com rótulo)")
g_auc_base = Gauge("credit_model_roc_auc_baseline", "ROC AUC na validação do treino")
g_recall = Gauge("credit_model_recall", "Recall da classe inadimplente")
g_f1 = Gauge("credit_model_f1", "F1 da classe inadimplente")
g_default = Gauge("credit_default_rate", "Taxa de inadimplência", ["kind"])
g_bad = Gauge("credit_bad_rate_approved", "Inadimplência entre clientes aprovados")
g_approval = Gauge("credit_approval_rate", "Taxa de aprovação")
g_psi = Gauge("credit_feature_psi", "PSI por feature vs. referência", ["feature"])
g_drift = Gauge("credit_feature_drift", "1 se drift confirmado (PSI>=0,10 e p<0,05)", ["feature"])
g_share = Gauge("credit_drift_share", "Fração de features com drift")
g_pred_psi = Gauge("credit_prediction_psi", "PSI do score do modelo")
g_alerts_active = Gauge("credit_alerts_active", "Alertas ativos no lote atual", ["type", "severity"])
c_alerts = Counter("credit_drift_alerts", "Alertas emitidos (acumulado)", ["type"])
c_batches = Counter("credit_batches", "Lotes recebidos por resultado do contrato", ["result"])
g_rows = Gauge("credit_batch_rows", "Linhas no lote atual")
g_latency = Gauge("credit_scoring_latency_ms", "Latência de scoring do lote (ms)")
g_stage = Gauge("credit_pipeline_stage_duration_seconds", "Duração da etapa do pipeline", ["stage"])
g_stage_ok = Gauge("credit_pipeline_stage_success", "1 se a etapa terminou com sucesso", ["stage"])
g_violations = Gauge("credit_contract_violations", "Violações no último lote bloqueado", ["batch"])

ALERT_TYPES = ["DATASET_DRIFT", "FEATURE_DRIFT", "PREDICTION_DRIFT", "CONCEPT_DRIFT", "CALIBRATION"]


def _load():
    summaries = {}
    for m in range(1, PRODUCTION_MONTHS + 1):
        p = REPORTS_DIR / "drift" / f"month_{m:02d}.json"
        if p.exists():
            summaries[m] = json.loads(p.read_text(encoding="utf-8"))
    logs = [json.loads(x) for x in LOG_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    last_run = logs[-1]["run_id"]
    logs = [e for e in logs if e["run_id"] == last_run]
    run = json.loads((REPORTS_DIR / "run_summary.json").read_text(encoding="utf-8"))
    return summaries, logs, run


def _batch_month(name: str) -> int | None:
    if name.startswith("production_month_"):
        return int(name.split("_")[2])
    return None


def publish(month: int, summaries: dict, logs: list, run: dict) -> None:
    s = summaries[month]
    perf = s["performance"]
    g_month.set(month)
    g_status.set(STATUS_CODE[s["status"]])
    g_auc.set(perf["roc_auc"])
    g_auc_base.set(run["baseline"]["roc_auc"])
    g_recall.set(perf["recall"])
    g_f1.set(perf["f1"])
    g_default.labels("real").set(perf["default_rate_real"])
    g_default.labels("prevista").set(perf["default_rate_predicted"])
    g_bad.set(perf["bad_rate_approved"])
    g_approval.set(perf["approval_rate"])
    for f in s["features"]:
        g_psi.labels(f["feature"]).set(f["psi"])
        g_drift.labels(f["feature"]).set(1 if f["drift_detected"] else 0)
    g_share.set(s["drift_share"])
    g_pred_psi.set(s["prediction_psi"])
    for t in ALERT_TYPES:
        for sev in ("critical", "high"):
            g_alerts_active.labels(t, sev).set(0)
    for a in s["alerts"]:
        g_alerts_active.labels(a["type"], a["severity"]).inc()
        c_alerts.labels(a["type"]).inc()
    g_rows.set(s["rows"])

    for e in logs:
        bm = _batch_month(e.get("batch", "")) if "batch" in e else None
        if e["event"] == "batch_scored" and bm == month:
            g_latency.set(e["latency_ms"])
        if e["event"] in ("batch_accepted", "batch_blocked") and bm == month:
            c_batches.labels("aceito" if e["event"] == "batch_accepted" else "bloqueado").inc()
            if e["event"] == "batch_blocked":
                g_violations.labels(e["batch"]).set(e["violations"])


def publish_pipeline(logs: list) -> None:
    for e in logs:
        if e["event"] == "stage_succeeded":
            g_stage.labels(e["stage"]).set(e.get("duration_s", 0))
            g_stage_ok.labels(e["stage"]).set(1)
        elif e["event"] == "stage_failed":
            g_stage_ok.labels(e["stage"]).set(0)
        elif e["event"] == "batch_blocked" and _batch_month(e["batch"]) is None:
            c_batches.labels("bloqueado").inc()
            g_violations.labels(e["batch"]).set(e["violations"])
        elif e["event"] == "batch_accepted" and _batch_month(e["batch"]) is None:
            c_batches.labels("aceito").inc()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--replay", type=float, default=0, help="segundos por mês (0 = só o último lote)")
    args = ap.parse_args()

    summaries, logs, run = _load()
    start_http_server(args.port)
    publish_pipeline(logs)
    print(f"Métricas em http://localhost:{args.port}/metrics")
    if not args.replay:
        for m in sorted(summaries):
            publish(m, summaries, logs, run)
        while True:
            time.sleep(3600)
    while True:
        for m in sorted(summaries):
            publish(m, summaries, logs, run)
            print(f"[replay] mês {m} publicado (status={summaries[m]['status']})", flush=True)
            time.sleep(args.replay)


if __name__ == "__main__":
    main()
