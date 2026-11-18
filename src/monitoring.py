"""Observabilidade: MLflow (rastreamento por lote), Prometheus (métricas) e dashboard HTML.

Fontes centralizadas:
  logs/pipeline.jsonl        eventos de todas as etapas (sucesso/falha, latência, volume)
  reports/drift/month_XX.json  resultados dos testes de drift e performance
  reports/causal_analysis.json efeitos causais por mecanismo
"""
from __future__ import annotations

import html
import json
from collections import Counter

import mlflow
import pandas as pd
import plotly.graph_objects as go
from prometheus_client import CollectorRegistry, Counter as PCounter, Gauge, write_to_textfile

from .config import (AUC_DROP_ALERT, DRIFT_SHARE_ALERT, FEATURES, MODEL_CARD_PATH, PSI_ALERT,
                     QUARANTINE_DIR, REPORTS_DIR)
from .logging_utils import RUN_ID, read_logs
from .tracking import setup_mlflow

# ---- paleta (dataviz reference palette, light) ----
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUE = [[0, "#f0efec"], [0.15, "#cde2fb"], [0.35, "#86b6ef"], [0.6, "#3987e5"], [0.8, "#1c5cab"],
            [1, "#0d366b"]]


# ----------------------------------------------------------------------------- MLflow
def log_month_to_mlflow(summary: dict, latency_ms: float) -> None:
    setup_mlflow("credit-scoring-production-monitoring")
    m = summary["month"]
    with mlflow.start_run(run_name=f"month_{m:02d}"):
        mlflow.set_tags({"pipeline_run_id": RUN_ID, "status": summary["status"], "month": m})
        perf = summary["performance"]
        mlflow.log_metrics({
            "rows": summary["rows"], "scoring_latency_ms": latency_ms,
            "drift_share": summary["drift_share"], "drifted_features": len(summary["drifted_features"]),
            "prediction_psi": summary["prediction_psi"], "alerts": len(summary["alerts"]),
            "roc_auc": perf["roc_auc"], "f1": perf["f1"], "recall": perf["recall"],
            "default_rate_real": perf["default_rate_real"],
            "default_rate_predicted": perf["default_rate_predicted"],
            "bad_rate_approved": perf["bad_rate_approved"], "auc_drop": summary["auc_drop"],
            **{f"psi_{f['feature']}": f["psi"] for f in summary["features"]},
        })
        mlflow.log_artifact(str(REPORTS_DIR.parent / summary["html_report"]), artifact_path="evidently")


# ----------------------------------------------------------------------------- Prometheus
def export_prometheus(summaries: list[dict], logs: list[dict]) -> str:
    reg = CollectorRegistry()
    g_psi = Gauge("credit_feature_psi", "PSI por feature (último lote)", ["feature"], registry=reg)
    g_auc = Gauge("credit_model_roc_auc", "ROC AUC do último lote rotulado", registry=reg)
    g_bad = Gauge("credit_bad_rate_approved", "Inadimplência entre aprovados", registry=reg)
    g_share = Gauge("credit_drift_share", "Fração de features com drift", registry=reg)
    c_alerts = PCounter("credit_drift_alerts", "Alertas de drift emitidos", ["type"], registry=reg)
    c_batches = PCounter("credit_batches", "Lotes processados por resultado", ["result"], registry=reg)
    g_dur = Gauge("credit_stage_duration_seconds", "Duração da etapa", ["stage"], registry=reg)

    last = summaries[-1]
    for f in last["features"]:
        g_psi.labels(f["feature"]).set(f["psi"])
    g_auc.set(last["performance"]["roc_auc"])
    g_bad.set(last["performance"]["bad_rate_approved"])
    g_share.set(last["drift_share"])
    for s in summaries:
        for a in s["alerts"]:
            c_alerts.labels(a["type"]).inc()
    for e in logs:
        if e["event"] == "batch_accepted":
            c_batches.labels("accepted").inc()
        elif e["event"] == "batch_blocked":
            c_batches.labels("blocked").inc()
        elif e["event"] == "stage_succeeded":
            g_dur.labels(e["stage"]).set(e.get("duration_s", 0))
    path = REPORTS_DIR / "metrics.prom"
    write_to_textfile(str(path), reg)
    return str(path)


# ----------------------------------------------------------------------------- Dashboard
def _layout(fig: go.Figure, title: str, height: int = 320, yfmt: str | None = None, ytitle: str = "") -> str:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)),
        height=height, margin=dict(l=56, r=48, t=56, b=44), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK2, size=12),
        legend=dict(orientation="h", y=-0.14, yanchor="top", x=0, xanchor="left", font=dict(color=INK2)),
        hoverlabel=dict(bgcolor="white", font=dict(color=INK)), hovermode=fig.layout.hovermode or "x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED),
                     tickformat=yfmt, title=dict(text=ytitle, font=dict(color=MUTED)))
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})


def _status_badge(status: str) -> str:
    icon, color, label = {"healthy": ("●", GOOD, "Saudável"), "warning": ("▲", WARNING, "Atenção"),
                          "critical": ("■", CRITICAL, "Crítico"), "success": ("✓", GOOD, "Sucesso"),
                          "error": ("✕", CRITICAL, "Falha"), "blocked": ("✕", CRITICAL, "Bloqueado"),
                          "accepted": ("✓", GOOD, "Aceito")}.get(status, ("•", MUTED, status))
    return f'<span class="badge"><span style="color:{color}">{icon}</span> {label}</span>'


def build_dashboard(summaries: list[dict], baseline: dict, causal: dict | None) -> str:
    logs = read_logs()
    run_logs = [e for e in logs if e["run_id"] == RUN_ID]
    months = [f"Mês {s['month']}" for s in summaries]
    perf = pd.DataFrame([s["performance"] for s in summaries])
    last = summaries[-1]
    blocked = [e for e in run_logs if e["event"] == "batch_blocked"]
    accepted = [e for e in run_logs if e["event"] == "batch_accepted"]
    n_alerts = sum(len(s["alerts"]) for s in summaries)
    charts = []

    # 1. AUC ao longo do tempo
    f = go.Figure()
    f.add_hline(y=baseline["roc_auc"], line=dict(color=MUTED, dash="dot", width=1),
                annotation_text=f"baseline {baseline['roc_auc']:.3f}", annotation_font_color=MUTED)
    f.add_hline(y=baseline["roc_auc"] - AUC_DROP_ALERT, line=dict(color=CRITICAL, dash="dash", width=1),
                annotation_text="limiar de alerta", annotation_font_color=CRITICAL,
                annotation_position="bottom right")
    f.add_trace(go.Scatter(x=months, y=perf.roc_auc, mode="lines+markers", name="ROC AUC",
                           line=dict(color=BLUE, width=2), marker=dict(size=9),
                           hovertemplate="%{y:.3f}<extra>ROC AUC</extra>"))
    charts.append(_layout(f, "ROC AUC por lote mensal", yfmt=".2f"))

    # 2. Taxas de inadimplência
    f = go.Figure()
    for col, name, color in [("default_rate_real", "Inadimplência real", BLUE),
                             ("default_rate_predicted", "Inadimplência prevista", ORANGE),
                             ("bad_rate_approved", "Inadimplência entre aprovados", AQUA)]:
        f.add_trace(go.Scatter(x=months, y=perf[col], mode="lines+markers", name=name,
                               line=dict(color=color, width=2), marker=dict(size=8),
                               hovertemplate="%{y:.1%}<extra>" + name + "</extra>"))
    f.update_layout(margin=dict(b=80))
    charts.append(_layout(f, "Inadimplência: real × prevista × aprovados", height=340, yfmt=".0%"))

    # 3. Heatmap PSI
    psi = pd.DataFrame({f"Mês {s['month']}": {x["feature"]: x["psi"] for x in s["features"]} for s in summaries})
    psi = psi.loc[psi.max(axis=1).sort_values(ascending=True).index]
    f = go.Figure(go.Heatmap(z=psi.values.clip(max=1.0), x=psi.columns, y=psi.index, colorscale=SEQ_BLUE,
                             zmin=0, zmax=1.0, xgap=2, ygap=2, customdata=psi.values,
                             text=[[f"{v:.2f}" if v >= PSI_ALERT else "" for v in row] for row in psi.values],
                             texttemplate="%{text}", textfont=dict(color="white", size=11),
                             colorbar=dict(title="PSI", tickvals=[0, 0.1, 0.25, 0.5, 1.0],
                                           ticktext=["0", "0,10", "0,25", "0,5", "≥1"]),
                             hovertemplate="%{y} · %{x}<br>PSI = %{customdata:.3f}<extra></extra>"))
    html_heat = _layout(f, "PSI por feature e mês (rótulo quando PSI ≥ 0,25)", height=420)
    charts.append(html_heat)

    # 4. Features com drift
    f = go.Figure(go.Bar(x=months, y=[len(s["drifted_features"]) for s in summaries], marker_color=BLUE,
                         marker_line=dict(color=SURFACE, width=2), name="features com drift",
                         customdata=[", ".join(s["drifted_features"]) or "—" for s in summaries],
                         hovertemplate="%{y} features<br>%{customdata}<extra></extra>"))
    f.add_hline(y=DRIFT_SHARE_ALERT * len(FEATURES), line=dict(color=CRITICAL, dash="dash", width=1),
                annotation_text="alerta de dataset drift (≥ 30% das features)", annotation_font_color=CRITICAL,
                annotation_position="top left")
    f.update_yaxes(range=[0, len(FEATURES) * 0.5])
    charts.append(_layout(f, "Features com drift confirmado (PSI ≥ 0,10 e p < 0,05)", ytitle="features"))

    # 5. Violações do contrato (lotes bloqueados nesta execução)
    viol = []
    for e in blocked:
        p = QUARANTINE_DIR / f"{e['batch']}_violations.csv"
        if p.exists():
            v = pd.read_csv(p)
            v["batch"] = e["batch"]
            viol.append(v)
    viol_html = ""
    if viol:
        v = pd.concat(viol)
        v = v[~v.batch.str.startswith("raw_")]  # foco nos lotes de produção / parceiro
        v = v.groupby("rule").violations.sum().sort_values()
        f = go.Figure(go.Bar(x=v.values, y=v.index, orientation="h", marker_color=CRITICAL,
                             marker_line=dict(color=SURFACE, width=2),
                             hovertemplate="%{y}: %{x} registros<extra></extra>"))
        f.update_layout(hovermode="closest")
        viol_html = _layout(f, "Registros barrados por regra do contrato (lotes de parceiro)", height=380)
        charts.append(viol_html)

    # 6. Causal
    causal_figs = []
    if causal:
        c = pd.DataFrame(causal["scenarios"]).iloc[1:]
        for metric, title, fmt in [("effect_roc_auc", "Efeito causal na ROC AUC (vs. controle)", ".3f"),
                                   ("effect_bad_rate_approved",
                                    "Efeito causal na inadimplência entre aprovados (p.p.)", ".1%"),
                                   ("effect_drift_share", "Efeito na fração de features com drift", ".0%")]:
            f = go.Figure(go.Bar(x=c[metric], y=c.scenario, orientation="h",
                                 marker_color=BLUE,
                                 marker_line=dict(color=SURFACE, width=2),
                                 hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
            f.update_layout(hovermode="closest")
            f.update_xaxes(tickformat=fmt)
            causal_figs.append(_layout(f, title, height=300))
        call = pd.DataFrame(causal["scenarios"])
        rows = "".join(
            f"<tr><td>{html.escape(r.scenario)}</td><td class='num'>{r.roc_auc_mean:.3f} ± {r.roc_auc_std:.3f}</td>"
            f"<td class='num'>{r.bad_rate_approved_mean:.1%}</td><td class='num'>{r.default_rate_real_mean:.1%}</td>"
            f"<td class='num'>{r.drift_share_mean:.0%}</td></tr>" for r in call.itertuples())
        causal_figs.append("<table><tr><th>Cenário</th><th>ROC AUC</th><th>Inad. aprovados</th>"
                           "<th>Inad. real</th><th>Features c/ drift</th></tr>" + rows + "</table>")

    # 7. Latência e volume de scoring
    sc = [e for e in run_logs if e["event"] == "batch_scored"]
    f = go.Figure(go.Bar(x=[e["batch"].replace("production_month_", "Mês ").replace("Mês 0", "Mês ") for e in sc],
                         y=[e["latency_ms"] for e in sc], marker_color=BLUE, marker_line=dict(color=SURFACE, width=2),
                         customdata=[[e["rows"], e["throughput_rows_s"]] for e in sc],
                         hovertemplate="%{y:.1f} ms · %{customdata[0]} linhas · %{customdata[1]:,} linhas/s<extra></extra>"))
    latency_html = _layout(f, "Latência de scoring por lote (ms)", height=300, ytitle="ms")

    # Tabelas
    stage_rows = "".join(
        f"<tr><td>{html.escape(e['stage'])}</td><td>{_status_badge(e['status'])}</td>"
        f"<td class='num'>{e.get('duration_s', 0):.2f}s</td><td>{html.escape(e['timestamp'])}</td></tr>"
        for e in run_logs if e["event"] in ("stage_succeeded", "stage_failed"))
    batch_rows = "".join(
        f"<tr><td>{html.escape(e['batch'])}</td><td>{_status_badge('accepted' if e['event']=='batch_accepted' else 'blocked')}</td>"
        f"<td class='num'>{e['rows']}</td><td class='num'>{e.get('violations', 0)}</td>"
        f"<td>{html.escape(', '.join(e.get('rules', [])[:6]))}</td></tr>"
        for e in run_logs if e["event"] in ("batch_accepted", "batch_blocked"))
    alert_rows = "".join(
        f"<tr><td>Mês {s['month']}</td><td>{_status_badge('critical' if a['severity']=='critical' else 'warning')}</td>"
        f"<td>{a['type']}</td><td>{html.escape(a['message'])}</td></tr>"
        for s in summaries for a in s["alerts"])
    month_rows = "".join(
        f"<tr><td>Mês {s['month']}</td><td>{_status_badge(s['status'])}</td><td class='num'>{s['rows']}</td>"
        f"<td class='num'>{s['performance']['roc_auc']:.3f}</td><td class='num'>{s['performance']['bad_rate_approved']:.1%}</td>"
        f"<td class='num'>{len(s['drifted_features'])}</td><td class='num'>{len(s['alerts'])}</td>"
        f"<td><a href='{s['html_report'].replace('reports/', '')}'>Evidently ↗</a></td></tr>" for s in summaries)
    alert_types = Counter(a["type"] for s in summaries for a in s["alerts"])

    def tile(label, value, sub="", status=None):
        badge = _status_badge(status) if status else ""
        return (f"<div class='tile'><div class='tl'>{label}</div><div class='tv'>{value}</div>"
                f"<div class='ts'>{badge} {sub}</div></div>")

    tiles = "".join([
        tile("Status do modelo (último lote)", {"critical": "Crítico", "warning": "Atenção",
                                                 "healthy": "Saudável"}[last["status"]], "", last["status"]),
        tile("ROC AUC atual", f"{last['performance']['roc_auc']:.3f}",
             f"baseline {baseline['roc_auc']:.3f} (−{last['auc_drop']:.3f})"),
        tile("Inadimplência entre aprovados", f"{last['performance']['bad_rate_approved']:.1%}",
             f"mês 1: {summaries[0]['performance']['bad_rate_approved']:.1%}"),
        tile("Lotes aceitos / bloqueados", f"{len(accepted)} / {len(blocked)}", "pelo contrato de dados"),
        tile("Alertas emitidos", str(n_alerts), " · ".join(f"{k} {v}" for k, v in alert_types.most_common(3))),
    ])

    plotly_js = __import__("plotly.offline", fromlist=["get_plotlyjs"]).get_plotlyjs()
    page = f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Credit Scoring — Monitoramento em Produção</title>
<script>{plotly_js}</script>
<style>
body{{margin:0;background:#f9f9f7;color:{INK};font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1240px;margin:0 auto;padding:28px 24px 64px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:36px 0 12px}}
.sub{{color:{INK2};margin:0 0 20px;font-size:14px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.tile{{background:{SURFACE};border:1px solid rgba(11,11,11,.10);border-radius:10px;padding:14px 16px}}
.tl{{font-size:12px;color:{INK2}}} .tv{{font-size:28px;font-weight:600;margin:6px 0 4px}} .ts{{font-size:12px;color:{MUTED}}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .card{{background:{SURFACE};border:1px solid rgba(11,11,11,.10);border-radius:10px;padding:6px;overflow:hidden}}
.full{{grid-column:1/-1}}
table{{width:100%;border-collapse:collapse;font-size:13px;background:{SURFACE};border-radius:10px;overflow:hidden}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid {GRID}}} th{{color:{INK2};font-weight:600;background:#f4f3ef}}
td.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}} .badge{{white-space:nowrap}}
a{{color:{BLUE}}} .note{{font-size:13px;color:{INK2};line-height:1.5}}
@media(max-width:860px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Credit Scoring — Observabilidade em Produção</h1>
<p class="sub">Execução <code>{RUN_ID}</code> · referência: amostra de validação do dataset de treino ·
{len(summaries)} lotes mensais · gerado automaticamente por <code>python -m src.pipeline</code></p>
<div class="tiles">{tiles}</div>

<h2>Saúde do modelo</h2>
<div class="grid"><div class="card">{charts[0]}</div><div class="card">{charts[1]}</div></div>
<h2>Data drift</h2>
<div class="grid"><div class="card">{charts[2]}</div><div class="card">{charts[3]}</div></div>
<h2>Resumo por lote</h2>
<table><tr><th>Lote</th><th>Status</th><th>Linhas</th><th>ROC AUC</th><th>Inad. aprovados</th><th>Features c/ drift</th><th>Alertas</th><th>Relatório</th></tr>{month_rows}</table>

<h2>Qualidade de dados (contratos)</h2>
<div class="grid"><div class="card">{viol_html}</div><div>
<table><tr><th>Lote</th><th>Resultado</th><th>Linhas</th><th>Violações</th><th>Regras</th></tr>{batch_rows}</table></div></div>

<h2>Saúde do pipeline</h2>
<div class="grid"><div><table><tr><th>Etapa</th><th>Status</th><th>Duração</th><th>Timestamp (UTC)</th></tr>{stage_rows}</table></div>
<div class="card">{latency_html}</div></div>

<h2>Alertas</h2>
<table><tr><th>Lote</th><th>Severidade</th><th>Tipo</th><th>Mensagem</th></tr>{alert_rows}</table>

<h2>Análise causal da degradação (mês 6, {causal['n_seeds'] if causal else 0} sementes)</h2>
<p class="note">Cada barra é uma intervenção que liga um único mecanismo sobre a mesma base. O estresse econômico
(concept drift) é o mecanismo que mais derruba a AUC e eleva a inadimplência entre aprovados — e é <b>invisível</b>
para os testes de data drift. Já a mudança de perfil de clientes dispara a maior parte dos alertas de drift sem
piorar a AUC: drift nas features não é sinônimo de degradação.</p>
<div class="grid">{''.join(f'<div class="card">{c}</div>' for c in causal_figs)}</div>
</main></body></html>"""
    out = REPORTS_DIR / "monitoring_dashboard.html"
    out.write_text(page, encoding="utf-8")
    return str(out)

