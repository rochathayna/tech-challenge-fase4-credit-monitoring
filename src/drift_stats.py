"""Testes estatísticos de drift implementados do zero (auditáveis), complementando o Evidently.

PSI  (Population Stability Index) — numéricas e categóricas
     PSI = Σ (atual% - ref%) * ln(atual% / ref%)
     < 0,10 estável | 0,10–0,25 alerta moderado | > 0,25 drift significativo
KS   (Kolmogorov-Smirnov 2 amostras) — numéricas; drift se p-valor < 0,05
Qui² — categóricas; drift se p-valor < 0,05
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import CATEGORICAL_FEATURES, KS_PVALUE, NUMERIC_FEATURES, PSI_ALERT, PSI_WARN

EPS = 1e-6


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref.dropna(), edges)[0] / max(len(ref.dropna()), 1)
    c = np.histogram(cur.dropna(), edges)[0] / max(len(cur.dropna()), 1)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.dropna()) | set(cur.dropna()))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_level(psi: float) -> str:
    return "drift" if psi >= PSI_ALERT else ("alerta" if psi >= PSI_WARN else "estavel")


def feature_drift_table(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_FEATURES:
        ks = stats.ks_2samp(ref[col], cur[col])
        psi = psi_numeric(ref[col], cur[col])
        rows.append({"feature": col, "type": "num", "psi": psi, "psi_level": psi_level(psi),
                     "test": "KS", "statistic": float(ks.statistic), "p_value": float(ks.pvalue),
                     "ref_mean": float(ref[col].mean()), "cur_mean": float(cur[col].mean())})
    for col in CATEGORICAL_FEATURES:
        cats = sorted(set(ref[col]) | set(cur[col]))
        table = np.array([ref[col].value_counts().reindex(cats, fill_value=0),
                          cur[col].value_counts().reindex(cats, fill_value=0)])
        chi = stats.chi2_contingency(table + 1)
        psi = psi_categorical(ref[col], cur[col])
        rows.append({"feature": col, "type": "cat", "psi": psi, "psi_level": psi_level(psi),
                     "test": "Qui2", "statistic": float(chi.statistic), "p_value": float(chi.pvalue),
                     "ref_mean": None, "cur_mean": None})
    df = pd.DataFrame(rows)
    # drift confirmado = PSI relevante (efeito) E teste significativo (evidência)
    df["drift_detected"] = (df["psi"] >= PSI_WARN) & (df["p_value"] < KS_PVALUE)
    return df.sort_values("psi", ascending=False).reset_index(drop=True)
