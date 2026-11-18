"""Monta o site estático do GitHub Pages em _site/ a partir das evidências versionadas.

    python -m src.build_site

Só usa a biblioteca padrão: o workflow .github/workflows/pages.yml roda este script e publica _site/.
"""
from __future__ import annotations

import html
import json
import shutil

from .config import REPORTS_DIR, ROOT

SITE = ROOT / "_site"
REPO = "https://github.com/rochathayna/tech-challenge-fase4-credit-monitoring"

INK, INK2, MUTED, GRID, SURFACE, PAGE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#f9f9f7"
BLUE, GOOD, WARNING, CRITICAL = "#2a78d6", "#0ca30c", "#fab219", "#d03b3b"


def _load(name):
    return json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))


def _status(s):
    icon, color, label = {"healthy": ("●", GOOD, "Saudável"), "warning": ("▲", WARNING, "Atenção"),
                          "critical": ("■", CRITICAL, "Crítico")}[s]
    return f'<span style="color:{color}">{icon}</span> {label}'


def build() -> None:
    shutil.rmtree(SITE, ignore_errors=True)
    (SITE / "reports").mkdir(parents=True)
    shutil.copytree(REPORTS_DIR / "drift", SITE / "reports" / "drift")
    for f in ("monitoring_dashboard.html", "metrics.prom", "data_contract_results.json",
              "causal_analysis.json", "run_summary.json"):
        shutil.copy(REPORTS_DIR / f, SITE / "reports" / f)
    shutil.copytree(ROOT / "docs" / "img", SITE / "img")
    shutil.copy(ROOT / "models" / "model_card.json", SITE / "reports" / "model_card.json")

    run = _load("run_summary.json")
    contract = _load("data_contract_results.json")
    causal = _load("causal_analysis.json")
    months = {m["month"]: json.loads((REPORTS_DIR / "drift" / f"month_{m['month']:02d}.json").read_text("utf-8"))
              for m in run["months"]}
    base = run["baseline"]
    first, last = months[1], months[max(months)]

    month_rows = "".join(
        f"<tr><td>Mês {m}</td><td>{_status(s['status'])}</td>"
        f"<td class=n>{s['performance']['roc_auc']:.3f}</td>"
        f"<td class=n>{s['performance']['default_rate_real']:.1%}</td>"
        f"<td class=n>{s['performance']['default_rate_predicted']:.1%}</td>"
        f"<td class=n>{s['performance']['bad_rate_approved']:.1%}</td>"
        f"<td>{', '.join(s['drifted_features']) or '—'}</td><td class=n>{len(s['alerts'])}</td>"
        f"<td><a href='reports/drift/month_{m:02d}.html'>abrir ↗</a></td></tr>"
        for m, s in months.items())

    viol_rows = ""
    for batch, rules in contract["violations"].items():
        total = sum(r["violations"] for r in rules)
        top = ", ".join(f"{r['rule']} ({r['violations']})" for r in rules[:6])
        viol_rows += f"<tr><td>{batch}</td><td>✕ bloqueado</td><td class=n>{total}</td><td>{html.escape(top)}</td></tr>"
    for batch, ok in contract["accepted"].items():
        if ok:
            viol_rows += f"<tr><td>{batch}</td><td>✓ aceito</td><td class=n>0</td><td>—</td></tr>"

    causal_rows = "".join(
        f"<tr><td>{html.escape(c['scenario'])}</td><td class=n>{c['roc_auc_mean']:.3f}</td>"
        f"<td class=n>{c['effect_roc_auc']:+.3f}</td><td class=n>{c['bad_rate_approved_mean']:.1%}</td>"
        f"<td class=n>{c['drift_share_mean']:.0%}</td></tr>" for c in causal["scenarios"])

    page = f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tech Challenge Fase 4 — Credit Scoring: drift, observabilidade e LGPD</title>
<style>
body{{margin:0;background:{PAGE};color:{INK};font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5}}
main{{max-width:1120px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-size:30px;line-height:1.2;margin:0 0 8px}} h2{{font-size:21px;margin:48px 0 6px}} h3{{font-size:16px;margin:0 0 4px}}
p{{margin:6px 0 12px}} .lead{{font-size:17px;color:{INK2};max-width:820px}} .muted{{color:{MUTED};font-size:13px}}
a{{color:{BLUE}}} code{{background:#f0efec;padding:1px 5px;border-radius:4px;font-size:13px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:24px 0}}
.tile{{background:{SURFACE};border:1px solid rgba(11,11,11,.1);border-radius:10px;padding:14px 16px}}
.tl{{font-size:12px;color:{INK2}}} .tv{{font-size:23px;font-weight:600;white-space:nowrap}} .ts{{font-size:12px;color:{MUTED}}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;margin-top:16px}}
.card{{background:{SURFACE};border:1px solid rgba(11,11,11,.1);border-radius:12px;padding:16px;display:flex;flex-direction:column}}
.card img{{width:100%;border-radius:8px;border:1px solid {GRID};margin-bottom:12px;aspect-ratio:16/10;object-fit:cover;object-position:top}}
.card p{{color:{INK2};font-size:14px;flex:1}} .btn{{display:inline-block;margin-top:8px;font-weight:600;text-decoration:none}}
table{{width:100%;border-collapse:collapse;font-size:14px;background:{SURFACE};border-radius:10px;overflow:hidden;margin-top:12px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid {GRID};vertical-align:top}} th{{background:#f4f3ef;color:{INK2};font-weight:600}}
td:first-child,td:nth-child(2){{white-space:nowrap}} td.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.steps li{{margin:4px 0}} .callout{{background:#eef4fc;border-left:3px solid {BLUE};padding:12px 16px;border-radius:6px;margin:16px 0}}
nav a{{margin-right:16px;font-size:14px}}
</style></head><body><main>
<nav><a href="#resultados">Resultados</a><a href="#evidencias">Evidências</a><a href="#contratos">Contratos</a>
<a href="#drift">Drift</a><a href="#causal">Análise causal</a><a href="#observabilidade">Observabilidade</a>
<a href="#governanca">LGPD</a><a href="{REPO}">Repositório ↗</a></nav>
<p class="muted" style="margin-top:28px">MLET · Tech Challenge · Fase 4</p>
<h1>Credit Scoring em produção: contratos de dados, drift, observabilidade e LGPD</h1>
<p class="lead">Um modelo XGBoost de risco de crédito sofre degradação silenciosa com inflação, alta de juros e
estresse econômico. Este projeto barra dados ruins na entrada, detecta o drift com Evidently AI, centraliza
logs e métricas (MLflow, Prometheus, Grafana) e documenta a governança sob a LGPD, com análise causal da degradação.</p>

<div class="tiles" id="resultados">
<div class="tile"><div class="tl">ROC AUC</div><div class="tv">{first['performance']['roc_auc']:.3f} → {last['performance']['roc_auc']:.3f}</div><div class="ts">baseline de validação {base['roc_auc']:.3f}</div></div>
<div class="tile"><div class="tl">Inadimplência entre aprovados</div><div class="tv">{first['performance']['bad_rate_approved']:.1%} → {last['performance']['bad_rate_approved']:.1%}</div><div class="ts">mês 1 → mês 6</div></div>
<div class="tile"><div class="tl">Features com drift (mês 6)</div><div class="tv">{len(last['drifted_features'])} / 11</div><div class="ts">PSI ≥ 0,10 e p &lt; 0,05</div></div>
<div class="tile"><div class="tl">Alertas emitidos</div><div class="tv">{sum(len(s['alerts']) for s in months.values())}</div><div class="ts">primeiro crítico no mês 3</div></div>
<div class="tile"><div class="tl">Lotes bloqueados pelo contrato</div><div class="tv">3</div><div class="ts">10 regras Pandera</div></div>
</div>

<h2 id="evidencias">Evidências: abra direto no navegador</h2>
<p class="muted">Tudo abaixo foi gerado automaticamente por <code>python -m src.pipeline</code>.</p>
<div class="cards">
<div class="card"><img src="img/dashboard.jpg" alt="Dashboard de monitoramento"><h3>Dashboard de monitoramento</h3>
<p>Saúde do modelo, PSI por feature e mês, contratos de dados, saúde e latência do pipeline, alertas e análise causal. Interativo.</p>
<a class="btn" href="reports/monitoring_dashboard.html">Abrir dashboard →</a></div>
<div class="card"><img src="img/evidently_ui.jpg" alt="Evidently UI"><h3>Evidently AI: relatórios e painel de monitoramento</h3>
<p>Relatório HTML do Evidently para cada mês de produção (tabela abaixo) e painel histórico no Evidently UI (captura ao lado): PSI de cada feature, KS nas numéricas, drift do score e do target, performance de classificação.</p>
<a class="btn" href="reports/drift/month_06.html">Abrir relatório do mês 6 (crítico) →</a>
<a class="btn" href="reports/drift/month_01.html">Abrir relatório do mês 1 (estável) →</a>
<a class="btn" href="img/evidently_ui.jpg">Ver painel do Evidently UI →</a></div>
<div class="card"><img src="img/mlflow_runs.jpg" alt="MLflow"><h3>MLflow: rastreamento</h3>
<p>Experimento de treino (modelo, model card, fairness) e experimento de monitoramento com um run por lote e mais de 25 métricas.
Captura da UI; para navegar, rode <code>make all &amp;&amp; make mlflow</code>.</p>
<a class="btn" href="img/mlflow_runs.jpg">Ver captura em tamanho real →</a></div>
<div class="card"><img src="img/prometheus_psi_auc.jpg" alt="Prometheus: séries de PSI e AUC"><h3>Prometheus: regras de alerta</h3>
<p>7 regras (concept drift, dataset drift, feature drift, prediction drift, calibração, inadimplência de aprovados, lote bloqueado) disparando no mês 6.</p>
<a class="btn" href="img/prometheus_alerts.jpg">Ver alertas disparando →</a>
<a class="btn" href="img/prometheus_psi_auc.jpg">Ver séries de PSI e AUC →</a>
<a class="btn" href="reports/metrics.prom">Ver métricas expostas →</a></div>
</div>

<h2 id="contratos">1 · Contratos de dados (Pandera)</h2>
<p>10 regras rígidas: sem nulos, idade 18–100, limites de renda, sem duplicatas, domínios fechados, coerência
emprego × idade e renda × comprometimento, e duas regras de privacidade (schema estrito rejeita CPF em claro;
<code>client_id</code> precisa ser pseudônimo SHA-256). Lote reprovado vai para a quarentena e não chega ao modelo.</p>
<table><tr><th>Lote</th><th>Resultado</th><th>Violações</th><th>Principais regras violadas</th></tr>{viol_rows}</table>
<p class="muted">Também bloqueado em produção: o 1º envio do mês 4 (arquivo corrompido do parceiro), com reenvio aceito.
<a href="reports/data_contract_results.json">JSON completo</a></p>

<h2 id="drift">2 · Simulação e detecção de drift (Evidently AI)</h2>
<p>Seis lotes mensais: mudança no perfil de clientes (covariate shift), inflação (renda e valor +15%), alta de juros
(+2,5 p.p.) e estresse econômico (concept drift). Referência = dataset de treino; produção = holdout nunca visto pelo modelo.</p>
<table><tr><th>Lote</th><th>Status</th><th>ROC AUC</th><th>Inad. real</th><th>Inad. prevista</th><th>Inad. aprovados</th>
<th>Features com drift</th><th>Alertas</th><th>Evidently</th></tr>{month_rows}</table>

<h2 id="causal">3 · Análise causal</h2>
<p>Intervenções <code>do(·)</code> ligando um mecanismo por vez no mês 6 ({causal['n_seeds']} sementes):</p>
<table><tr><th>Cenário</th><th>ROC AUC</th><th>Efeito na AUC</th><th>Inad. aprovados</th><th>Features c/ drift</th></tr>{causal_rows}</table>
<div class="callout"><b>Conclusão:</b> o estresse econômico (concept drift) é o que mais derruba o modelo e é <b>invisível</b>
para os testes de data drift (0% de features com drift). Já a mudança de perfil de clientes acende a maior parte dos
alertas de drift sem piorar a AUC. Drift não é sinônimo de degradação, e monitorar só features não basta.</div>

<h2 id="observabilidade">4 · Observabilidade: como rodar a stack completa</h2>
<ol class="steps">
<li><code>pip install -r requirements.txt</code> e <code>make all</code>: executa o pipeline ponta a ponta (~30 s)</li>
<li><code>make mlflow</code>: UI do MLflow em <code>http://127.0.0.1:5000</code></li>
<li><code>make evidently-ui</code>: Evidently UI com o histórico mensal em <code>http://127.0.0.1:8001</code></li>
<li><code>cd monitoring &amp;&amp; docker compose up</code>: exporter + Prometheus (<code>:9090</code>) + Grafana (<code>:3000</code>) com dashboard provisionado; a produção é "reproduzida" um mês a cada 15 s</li>
</ol>

<h2 id="governanca">5 · Governança e LGPD</h2>
<p>CPF pseudonimizado com HMAC-SHA256 na ingestão e descartado; dados sensíveis (art. 5º, II) não coletados;
base legal: proteção do crédito (art. 7º, X) e procedimentos preliminares de contrato (art. 7º, V); revisão humana e
explicação de decisões automatizadas (art. 20); plano de retenção por tipo de dado; métricas de viés por idade e moradia
no <a href="reports/model_card.json">model card</a>.
<a href="{REPO}#etapa-4--governança-e-lgpd">Documentação completa no README →</a></p>

<p class="muted" style="margin-top:48px">Código, histórico de commits e documentação: <a href="{REPO}">{REPO}</a></p>
</main></body></html>"""
    (SITE / "index.html").write_text(page, encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    print(f"site gerado em {SITE}")


if __name__ == "__main__":
    build()
