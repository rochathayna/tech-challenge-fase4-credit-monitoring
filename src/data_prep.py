"""Ingestão bruta, pseudonimização (LGPD) e separação Referência x Produção.

O dataset público não traz identificadores. Para exercitar o fluxo real de uma fintech,
a ingestão simula a chegada de um CPF **sintético** (gerado aleatoriamente, não pertence a
ninguém). Esse CPF é imediatamente substituído por um pseudônimo HMAC-SHA256 e descartado —
nenhum dado identificável chega ao treino, aos relatórios ou aos logs.
"""
from __future__ import annotations

import hashlib
import hmac
import os

import numpy as np
import pandas as pd

from .config import (FEATURES, HOLDOUT_PATH, ID_COL, PROCESSED_DIR, RANDOM_STATE, RAW_PATH,
                     REFERENCE_PATH, TARGET)

# Em produção a chave vem de um cofre (Vault / Secret Manager). Nunca versionar a chave real.
PII_KEY = os.environ.get("PII_HMAC_KEY", "dev-only-key-troque-em-producao").encode()


def _fake_cpf(rng: np.random.Generator, n: int) -> list[str]:
    """CPF sintético com dígitos verificadores válidos (só para simular PII na ingestão)."""
    out = []
    for base in rng.integers(0, 10, size=(n, 9)):
        digits = list(base)
        for k in (10, 11):
            s = sum(d * w for d, w in zip(digits, range(k, 1, -1)))
            dv = (s * 10) % 11
            digits.append(0 if dv == 10 else dv)
        out.append("".join(map(str, digits)))
    return out


def pseudonymize(cpf: str) -> str:
    return hmac.new(PII_KEY, cpf.encode(), hashlib.sha256).hexdigest()


def load_raw(with_pii: bool = True) -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    if with_pii:
        rng = np.random.default_rng(RANDOM_STATE)
        df.insert(0, "cpf", _fake_cpf(rng, len(df)))
    return df


def apply_privacy_by_design(df: pd.DataFrame) -> pd.DataFrame:
    """Pseudonimiza o CPF e remove a PII direta (minimização)."""
    df = df.copy()
    if "cpf" in df.columns:
        df[ID_COL] = df["cpf"].map(pseudonymize)
        df = df.drop(columns=["cpf"])
    cols = [ID_COL] + FEATURES + ([TARGET] if TARGET in df.columns else [])
    return df[cols]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza que gera o Dataset de Referência — espelha as regras do contrato."""
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df.drop_duplicates(subset=FEATURES)
    df = df[(df.person_age >= 18) & (df.person_age <= 100)]
    df = df[(df.person_emp_length >= 0) & (df.person_emp_length <= 60)]
    df = df[df.person_emp_length <= df.person_age - 14]
    df = df[(df.person_income > 0) & (df.person_income <= 2_000_000)]
    df = df[(df.loan_percent_income - df.loan_amnt / df.person_income).abs() <= 0.02]
    df = df.astype({"person_age": "int64", "person_income": "float64", "loan_amnt": "float64",
                    "cb_person_cred_hist_length": "int64", TARGET: "int64"})
    return df.reset_index(drop=True)


def split_reference_holdout(df: pd.DataFrame, ref_frac: float = 0.6):
    """Referência (treino) x Holdout (pool de onde nascem os lotes de produção)."""
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    cut = int(len(df) * ref_frac)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def build_datasets() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    pseudo = apply_privacy_by_design(raw)
    cleaned = clean(pseudo)
    reference, holdout = split_reference_holdout(cleaned)
    reference.to_csv(REFERENCE_PATH, index=False)
    holdout.to_csv(HOLDOUT_PATH, index=False)
    return {"raw": raw, "pseudo": pseudo, "reference": reference, "holdout": holdout,
            "removed_by_cleaning": len(pseudo) - len(cleaned)}
