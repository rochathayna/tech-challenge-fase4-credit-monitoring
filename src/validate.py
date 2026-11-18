"""Gate de ingestão: todo lote passa pelo contrato antes de chegar ao modelo.

Lote reprovado => bloqueado, enviado para data/quarantine/ com o relatório de falhas,
e um evento `batch_blocked` é registrado nos logs centralizados.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import QUARANTINE_DIR, RANDOM_STATE, REPORTS_DIR
from .contracts import DataContractError, summarize_failures, validate_batch
from .logging_utils import log_event


def ingest(df: pd.DataFrame, batch_name: str, require_target: bool = True) -> pd.DataFrame | None:
    """Retorna o lote validado ou None se foi bloqueado."""
    try:
        validated = validate_batch(df, require_target=require_target)
    except DataContractError as err:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        summary = summarize_failures(err.failure_cases)
        df.to_csv(QUARANTINE_DIR / f"{batch_name}.csv", index=False)
        summary.to_csv(QUARANTINE_DIR / f"{batch_name}_violations.csv", index=False)
        log_event("validation", "batch_blocked", "error", batch=batch_name, rows=len(df),
                  violations=int(summary.violations.sum()), rules_violated=int(summary.shape[0]),
                  rules=summary.rule.unique().tolist())
        return None
    log_event("validation", "batch_accepted", "success", batch=batch_name, rows=len(df))
    return validated


def make_corrupted_batch(clean_df: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    """Lote propositalmente sujo — simula falha de integração com o parceiro de dados."""
    rng = np.random.default_rng(RANDOM_STATE)
    bad = clean_df.sample(n, random_state=RANDOM_STATE).reset_index(drop=True).copy()
    idx = rng.permutation(n)
    bad.loc[idx[:25], "person_income"] = np.nan                   # R1/R3 renda nula
    bad.loc[idx[25:45], "person_age"] = rng.integers(12, 18, 20)   # R2 menor de idade
    bad.loc[idx[45:50], "person_age"] = 144                        # R2 erro de digitação
    bad.loc[idx[50:60], "person_income"] = -1000.0                 # R3 renda negativa
    bad.loc[idx[60:70], "loan_intent"] = "CRYPTO"                  # R5 domínio desconhecido
    bad.loc[idx[70:80], "person_emp_length"] = 123.0               # R6 incoerente
    bad.loc[idx[80:90], "loan_int_rate"] = 99.0                    # R7 taxa fora do range
    bad = pd.concat([bad, bad.iloc[:15]], ignore_index=True)       # R4 duplicatas
    bad["cpf"] = "00000000000"                                     # R9 PII em claro vazando
    return bad


def run_contract_demo(reference: pd.DataFrame, raw_pseudo: pd.DataFrame) -> dict:
    """Etapa 1 — evidências: dado bruto reprovado, referência aprovada, lote corrompido bloqueado."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    raw_ok = ingest(raw_pseudo, "raw_kaggle_ingest")
    results["raw_kaggle_ingest"] = raw_ok is not None

    ref_ok = ingest(reference, "reference_dataset")
    results["reference_dataset"] = ref_ok is not None

    bad = make_corrupted_batch(reference)
    bad_ok = ingest(bad, "corrupted_partner_batch")
    results["corrupted_partner_batch"] = bad_ok is not None

    summaries = {}
    for name in ("raw_kaggle_ingest", "corrupted_partner_batch"):
        p = QUARANTINE_DIR / f"{name}_violations.csv"
        if p.exists():
            summaries[name] = pd.read_csv(p).to_dict(orient="records")
    (REPORTS_DIR / "data_contract_results.json").write_text(
        json.dumps({"accepted": results, "violations": summaries}, indent=2, ensure_ascii=False), encoding="utf-8")
    return results
