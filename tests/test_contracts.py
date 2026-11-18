import numpy as np
import pandas as pd
import pytest

from src.contracts import DataContractError, summarize_failures, validate_batch
from src.data_prep import pseudonymize
from src.validate import make_corrupted_batch


@pytest.fixture
def clean_batch() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 50
    income = rng.integers(20_000, 150_000, n).astype(float)
    amnt = rng.integers(1_000, 20_000, n).astype(float)
    return pd.DataFrame({
        "client_id": [pseudonymize(f"{i:011d}") for i in range(n)],
        "person_age": rng.integers(21, 70, n),
        "person_income": income,
        "person_emp_length": rng.integers(0, 6, n).astype(float),
        "loan_amnt": amnt,
        "loan_int_rate": rng.uniform(5, 20, n).round(2),
        "loan_percent_income": (amnt / income).round(2),
        "cb_person_cred_hist_length": rng.integers(2, 20, n),
        "person_home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
        "loan_intent": rng.choice(["EDUCATION", "MEDICAL", "PERSONAL"], n),
        "loan_grade": rng.choice(list("ABCD"), n),
        "cb_person_default_on_file": rng.choice(["Y", "N"], n),
        "loan_status": rng.integers(0, 2, n),
    })


def test_clean_batch_passes(clean_batch):
    assert len(validate_batch(clean_batch)) == len(clean_batch)


def _rules(df, **kw):
    with pytest.raises(DataContractError) as exc:
        validate_batch(df, **kw)
    return set(summarize_failures(exc.value.failure_cases).rule)


def test_null_income_blocked(clean_batch):
    clean_batch.loc[0, "person_income"] = np.nan
    assert "R1_sem_nulos" in _rules(clean_batch)


def test_minor_blocked(clean_batch):
    clean_batch.loc[0, "person_age"] = 16
    assert "R2_idade_18_100" in _rules(clean_batch)


def test_duplicates_blocked(clean_batch):
    dup = pd.concat([clean_batch, clean_batch.iloc[[0]]], ignore_index=True)
    assert {"R4_client_id_duplicado", "R4_registro_duplicado"} <= _rules(dup)


def test_unknown_category_blocked(clean_batch):
    clean_batch.loc[0, "loan_intent"] = "CRYPTO"
    assert "R5_finalidade" in _rules(clean_batch)


def test_raw_pii_column_blocked(clean_batch):
    clean_batch["cpf"] = "12345678909"
    assert any(r.startswith("R9_coluna_nao_prevista") for r in _rules(clean_batch))


def test_non_pseudonymized_id_blocked(clean_batch):
    clean_batch.loc[0, "client_id"] = "12345678909"
    assert "R10_pseudonimo_sha256" in _rules(clean_batch)


def test_scoring_contract_without_target(clean_batch):
    assert len(validate_batch(clean_batch.drop(columns=["loan_status"]), require_target=False)) == 50


def test_corrupted_batch_hits_many_rules(clean_batch):
    rules = _rules(make_corrupted_batch(clean_batch, n=50))
    assert len(rules) >= 6
