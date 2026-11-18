"""Simulação de 6 meses de produção com degradação econômica progressiva.

Mecanismo gerador (intensidade t = (mês-1)/5: mês 1 = estável, mês 6 = degradação máxima):

  [DATA DRIFT — covariate shift, P(X) muda e P(y|X) é preservada]
  D1  Novo perfil de clientes: reamostragem ponderada favorecendo propostas mais arriscadas
      (maior comprometimento de renda, juros altos, clientes mais jovens e de aluguel).
      Como as linhas são reais, a relação X -> y continua verdadeira.

  [DATA DRIFT nominal — o "número" muda, o risco real não acompanha]
  D2  Inflação: renda nominal +15%*t e valor do empréstimo +15%*t (o poder de compra não sobe).
  D3  Aperto monetário: taxa de juros +2,5 p.p.*t (Selic em alta repassada ao crédito).

  [CONCEPT DRIFT — P(y|X) muda]
  C1  Estresse econômico (desemprego / endividamento): clientes adimplentes com alto
      comprometimento de renda passam a inadimplir com probabilidade crescente em t.
      Mesmo X, mais default -> o modelo treinado no passado passa a errar.

Cada lote é gerado com uma semente própria e salvo em data/production/month_XX.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HOLDOUT_PATH, PRODUCTION_DIR, PRODUCTION_MONTHS, RANDOM_STATE, TARGET

BATCH_SIZE = 1500


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def covariate_shift(pool: pd.DataFrame, t: float, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """D1 — muda o mix de clientes sem alterar nenhum valor individual."""
    risk_profile = (_z(pool.loan_percent_income) + 0.8 * _z(pool.loan_int_rate)
                    - 0.6 * _z(pool.person_age) + 0.8 * (pool.person_home_ownership == "RENT"))
    w = np.exp(0.5 * t * risk_profile.to_numpy())
    idx = rng.choice(len(pool), size=n, replace=False, p=w / w.sum())
    return pool.iloc[idx].reset_index(drop=True).copy()


def nominal_shift(df: pd.DataFrame, t: float) -> pd.DataFrame:
    """D2 + D3 — inflação e juros."""
    df = df.copy()
    infl = 1 + 0.15 * t
    df["person_income"] = (df["person_income"] * infl).round(0)
    df["loan_amnt"] = (df["loan_amnt"] * infl).clip(upper=35_000).round(0)
    df["loan_percent_income"] = (df["loan_amnt"] / df["person_income"]).round(2).clip(0, 1)
    df["loan_int_rate"] = (df["loan_int_rate"] + 2.5 * t).clip(upper=35).round(2)
    return df


def concept_shift(df: pd.DataFrame, t: float, rng: np.random.Generator) -> pd.DataFrame:
    """C1 — mesma proposta, maior chance de default sob estresse econômico."""
    df = df.copy()
    stress = ((df["loan_percent_income"] - 0.10) / 0.25).clip(0, 1) * 0.7 \
        + ((df["loan_int_rate"] - 10) / 10).clip(0, 1) * 0.3
    p_flip = 0.35 * t * stress
    flip = (df[TARGET] == 0) & (rng.random(len(df)) < p_flip)
    df.loc[flip, TARGET] = 1
    return df


def generate_month(pool: pd.DataFrame, month: int, *, data_drift: bool = True, nominal: bool = True,
                   concept: bool = True, n: int = BATCH_SIZE, seed_offset: int = 0) -> pd.DataFrame:
    t = (month - 1) / (PRODUCTION_MONTHS - 1)
    rng = np.random.default_rng(RANDOM_STATE + 100 * month + seed_offset)
    if data_drift:
        df = covariate_shift(pool, t, n, rng)
    else:
        df = pool.sample(n, replace=False, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    if nominal:
        df = nominal_shift(df, t)
    if concept:
        df = concept_shift(df, t, rng)
    # cada linha de produção é uma nova proposta: novo pseudônimo de proposta
    df["client_id"] = [f"{h[:56]}{month:02d}{i:06d}" for i, h in enumerate(df["client_id"])]
    return df


def build_production_batches() -> dict[int, pd.DataFrame]:
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(HOLDOUT_PATH)
    batches = {}
    for m in range(1, PRODUCTION_MONTHS + 1):
        df = generate_month(pool, m)
        df.to_csv(PRODUCTION_DIR / f"month_{m:02d}.csv", index=False)
        batches[m] = df
    return batches
