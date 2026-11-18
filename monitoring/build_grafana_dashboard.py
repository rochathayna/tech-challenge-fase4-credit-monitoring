"""Gera monitoring/grafana/dashboards/credit-monitoring.json (dashboard provisionado do Grafana).

    python monitoring/build_grafana_dashboard.py
"""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
BLUE, ORANGE, AQUA, VIOLET, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
_id = 0


def _next():
    global _id
    _id += 1
    return _id


def target(expr, legend="", instant=False, ref="A"):
    t = {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref}
    if instant:
        t.update({"instant": True, "range": False, "format": "table"})
    return t


def stat(title, expr, x, y, w=4, h=4, unit="none", decimals=None, steps=None, mappings=None, desc=""):
    return {
        "id": _next(), "type": "stat", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [target(expr)],
        "options": {"colorMode": "background", "graphMode": "area", "justifyMode": "center",
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "textMode": "value"},
        "fieldConfig": {"defaults": {
            "unit": unit, "decimals": decimals, "mappings": mappings or [],
            "thresholds": {"mode": "absolute", "steps": steps or [{"color": BLUE, "value": None}]},
            "color": {"mode": "thresholds"}}, "overrides": []},
    }


def timeseries(title, targets, x, y, w=12, h=8, unit="none", decimals=None, thresholds=None, colors=None,
               desc="", minv=None, maxv=None):
    overrides = [{"matcher": {"id": "byName", "options": name},
                  "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": c}}]}
                 for name, c in (colors or {}).items()]
    return {
        "id": _next(), "type": "timeseries", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
        "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
        "fieldConfig": {"defaults": {
            "unit": unit, "decimals": decimals, "min": minv, "max": maxv,
            "custom": {"drawStyle": "line", "lineWidth": 2, "fillOpacity": 0, "pointSize": 6,
                       "showPoints": "auto", "lineInterpolation": "stepAfter", "spanNulls": True,
                       "thresholdsStyle": {"mode": "dashed" if thresholds else "off"}},
            "color": {"mode": "palette-classic"},
            "thresholds": {"mode": "absolute", "steps": thresholds or [{"color": "transparent", "value": None}]},
        }, "overrides": overrides},
    }


def bargauge(title, expr, legend, x, y, w=12, h=8, unit="none", steps=None, desc="", maxv=None):
    return {
        "id": _next(), "type": "bargauge", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [target(expr, legend)],
        "options": {"displayMode": "basic", "orientation": "horizontal", "showUnfilled": True,
                    "valueMode": "color", "namePlacement": "left", "sizing": "auto",
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
        "fieldConfig": {"defaults": {"unit": unit, "min": 0, "max": maxv, "decimals": 2,
                                     "thresholds": {"mode": "absolute", "steps": steps},
                                     "color": {"mode": "thresholds"}}, "overrides": []},
    }


def row(title, y):
    return {"id": _next(), "type": "row", "title": title, "collapsed": False,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}


PSI_STEPS = [{"color": GOOD, "value": None}, {"color": WARNING, "value": 0.10}, {"color": CRITICAL, "value": 0.25}]
STATUS_MAP = [{"type": "value", "options": {
    "0": {"text": "Saudável", "color": GOOD, "index": 0},
    "1": {"text": "Atenção", "color": WARNING, "index": 1},
    "2": {"text": "Crítico", "color": CRITICAL, "index": 2}}}]

panels = [
    row("Saúde do modelo", 0),
    stat("Status do modelo", "credit_model_status", 0, 1, mappings=STATUS_MAP,
         steps=[{"color": GOOD, "value": None}, {"color": WARNING, "value": 1}, {"color": CRITICAL, "value": 2}]),
    stat("Lote (mês)", "credit_current_month", 4, 1, steps=[{"color": BLUE, "value": None}]),
    stat("ROC AUC", "credit_model_roc_auc", 8, 1, decimals=3,
         steps=[{"color": CRITICAL, "value": None}, {"color": WARNING, "value": 0.90}, {"color": GOOD, "value": 0.93}]),
    stat("Inadimplência entre aprovados", "credit_bad_rate_approved", 12, 1, unit="percentunit", decimals=1,
         steps=[{"color": GOOD, "value": None}, {"color": WARNING, "value": 0.10}, {"color": CRITICAL, "value": 0.15}]),
    stat("Features com drift", "sum(credit_feature_drift)", 16, 1,
         steps=[{"color": GOOD, "value": None}, {"color": WARNING, "value": 1}, {"color": CRITICAL, "value": 4}]),
    stat("Alertas ativos (Prometheus)", 'count(ALERTS{alertstate="firing"}) or vector(0)', 20, 1,
         steps=[{"color": GOOD, "value": None}, {"color": CRITICAL, "value": 1}]),
    timeseries("ROC AUC × baseline × limiar de alerta", [
        target("credit_model_roc_auc", "ROC AUC", ref="A"),
        target("credit_model_roc_auc_baseline", "baseline", ref="B"),
        target("credit_model_roc_auc_baseline - 0.05", "limiar de alerta", ref="C")],
        0, 5, unit="none", decimals=3, minv=0.85, maxv=0.96,
        colors={"ROC AUC": BLUE, "baseline": "#898781", "limiar de alerta": CRITICAL}),
    timeseries("Inadimplência: real × prevista × entre aprovados", [
        target('credit_default_rate{kind="real"}', "real", ref="A"),
        target('credit_default_rate{kind="prevista"}', "prevista pelo modelo", ref="B"),
        target("credit_bad_rate_approved", "entre aprovados", ref="C")],
        12, 5, unit="percentunit", decimals=1,
        colors={"real": BLUE, "prevista pelo modelo": ORANGE, "entre aprovados": AQUA}),
    row("Data drift", 13),
    timeseries("PSI por feature (0,10 atenção · 0,25 drift)", [target("credit_feature_psi", "{{feature}}")],
               0, 14, w=14, h=9, decimals=2, thresholds=PSI_STEPS,
               desc="Population Stability Index de cada feature vs. dataset de referência"),
    bargauge("PSI atual por feature", "sort_desc(credit_feature_psi)", "{{feature}}", 14, 14, w=10, h=9,
             steps=PSI_STEPS, maxv=1),
    timeseries("Drift do score do modelo e fração de features com drift", [
        target("credit_prediction_psi", "PSI do score", ref="A"),
        target("credit_drift_share", "fração de features com drift", ref="B")],
        0, 23, decimals=2, colors={"PSI do score": VIOLET, "fração de features com drift": ORANGE}),
    {
        "id": _next(), "type": "table", "title": "Alertas disparando no Prometheus", "datasource": DS,
        "gridPos": {"x": 12, "y": 23, "w": 12, "h": 8},
        "targets": [target('ALERTS{alertstate="firing"}', instant=True)],
        "transformations": [{"id": "organize", "options": {
            "excludeByName": {"Time": True, "Value": True, "__name__": True, "alertstate": True,
                              "instance": True, "job": True, "kind": True},
            "renameByName": {"alertname": "Alerta", "severity": "Severidade", "feature": "Feature"},
            "indexByName": {"alertname": 0, "severity": 1, "feature": 2}}}],
        "fieldConfig": {"defaults": {}, "overrides": [{"matcher": {"id": "byName", "options": "Severidade"},
            "properties": [{"id": "custom.cellOptions", "value": {"type": "color-text"}},
                           {"id": "mappings", "value": [{"type": "value", "options": {
                               "critical": {"color": CRITICAL, "text": "crítico"},
                               "high": {"color": WARNING, "text": "alto"}}}]}]}]},
        "options": {"showHeader": True},
    },
    row("Pipeline, qualidade de dados e infraestrutura", 31),
    stat("Lotes aceitos", 'sum(credit_batches_total{result="aceito"})', 0, 32,
         steps=[{"color": GOOD, "value": None}]),
    stat("Lotes bloqueados (contrato)", 'sum(credit_batches_total{result="bloqueado"})', 4, 32,
         steps=[{"color": GOOD, "value": None}, {"color": CRITICAL, "value": 1}]),
    stat("Linhas no lote", "credit_batch_rows", 8, 32, steps=[{"color": BLUE, "value": None}]),
    stat("Latência de scoring", "credit_scoring_latency_ms", 12, 32, unit="ms", decimals=1,
         steps=[{"color": GOOD, "value": None}, {"color": WARNING, "value": 200}, {"color": CRITICAL, "value": 500}]),
    stat("Etapas com sucesso", "sum(credit_pipeline_stage_success)", 16, 32,
         steps=[{"color": GOOD, "value": None}]),
    stat("Alertas emitidos (acumulado)", "sum(credit_drift_alerts_total)", 20, 32,
         steps=[{"color": BLUE, "value": None}]),
    bargauge("Duração por etapa do pipeline (s)", "credit_pipeline_stage_duration_seconds", "{{stage}}",
             0, 36, unit="s", steps=[{"color": BLUE, "value": None}]),
    bargauge("Violações nos lotes bloqueados", "credit_contract_violations", "{{batch}}", 12, 36,
             steps=[{"color": CRITICAL, "value": None}]),
]

dashboard = {
    "uid": "credit-scoring-obs", "title": "Credit Scoring — Observabilidade",
    "description": "Tech Challenge Fase 4 — saúde do modelo, data drift, alertas e pipeline",
    "tags": ["credit-scoring", "drift", "mlops"], "timezone": "browser", "schemaVersion": 39,
    "refresh": "5s", "time": {"from": "now-5m", "to": "now"}, "editable": True,
    "graphTooltip": 1, "panels": panels, "templating": {"list": []}, "annotations": {"list": []},
}

out = Path(__file__).parent / "grafana" / "dashboards" / "credit-monitoring.json"
out.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"dashboard salvo em {out} ({len(panels)} painéis)")
