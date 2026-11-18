"""Análise causal da degradação — intervenções controladas no processo gerador.

Pergunta: *por que* o modelo degradou? Correlação no dashboard não basta; precisamos
separar o efeito de cada mecanismo. Como o processo gerador é conhecido (simulação),
aplicamos intervenções do tipo do(·) no mês 6, ligando um mecanismo por vez sobre a
mesma base, com várias sementes (Monte Carlo) para obter média e intervalo:

  controle         nenhum mecanismo (lote estável)
  do(D1)           só mudança no perfil de clientes (covariate shift)
  do(D2+D3)        só inflação + juros (mudança nominal)
  do(C1)           só estresse econômico (concept drift)
  completo         D1 + D2 + D3 + C1 (cenário observado em produção)

Efeito causal de cada mecanismo = métrica(intervenção) − métrica(controle).
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from .config import FEATURES, HOLDOUT_PATH, MODEL_PATH, PRODUCTION_MONTHS, REPORTS_DIR, TARGET
from .drift_stats import feature_drift_table
from .simulate_drift import generate_month
from .train import classification_metrics, score

SCENARIOS = {
    "controle": dict(data_drift=False, nominal=False, concept=False),
    "do(D1) perfil de clientes": dict(data_drift=True, nominal=False, concept=False),
    "do(D2+D3) inflação e juros": dict(data_drift=False, nominal=True, concept=False),
    "do(C1) estresse econômico": dict(data_drift=False, nominal=False, concept=True),
    "completo (produção)": dict(data_drift=True, nominal=True, concept=True),
}


def run_causal_analysis(reference_scored: pd.DataFrame, n_seeds: int = 20) -> dict:
    model = joblib.load(MODEL_PATH)
    pool = pd.read_csv(HOLDOUT_PATH)
    results = []
    for name, flags in SCENARIOS.items():
        rows = []
        for seed in range(n_seeds):
            df = score(model, generate_month(pool, PRODUCTION_MONTHS, seed_offset=1000 + seed, **flags))
            m = classification_metrics(df[TARGET], df["prediction_proba"])
            appr = df[df.prediction == 0]
            m["bad_rate_approved"] = float(appr[TARGET].mean())
            m["drift_share"] = float(feature_drift_table(reference_scored, df).drift_detected.mean())
            rows.append(m)
        r = pd.DataFrame(rows)
        results.append({"scenario": name, **{f"{k}_mean": float(r[k].mean()) for k in r.columns},
                        **{f"{k}_std": float(r[k].std()) for k in ("roc_auc", "bad_rate_approved")}})
    res = pd.DataFrame(results)
    ctrl = res.iloc[0]
    for k in ("roc_auc", "bad_rate_approved", "default_rate_real", "drift_share"):
        res[f"effect_{k}"] = res[f"{k}_mean"] - ctrl[f"{k}_mean"]

    out = {"n_seeds": n_seeds, "month": PRODUCTION_MONTHS, "features": FEATURES,
           "scenarios": json.loads(res.to_json(orient="records"))}
    (REPORTS_DIR / "causal_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
    return out
