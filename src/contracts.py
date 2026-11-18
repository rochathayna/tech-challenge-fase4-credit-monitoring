"""Contrato de dados (Pandera) — a porta de entrada do modelo.

Regras rígidas (todas bloqueantes):
  R1  Sem nulos em nenhuma coluna
  R2  Idade entre 18 e 100 anos
  R3  Renda > 0 e <= 2.000.000 (limite de renda)
  R4  Sem duplicatas (client_id único e nenhum registro de proposta repetido)
  R5  Domínios categóricos fechados (moradia, finalidade, grade, histórico)
  R6  Tempo de emprego entre 0 e 60 anos e coerente com a idade (emp_length <= idade - 14)
  R7  Valor do empréstimo entre 500 e 35.000; taxa de juros entre 0 e 35% a.a.
  R8  loan_percent_income coerente com loan_amnt / person_income (tolerância 0,02)
  R9  Privacidade: schema estrito — qualquer coluna extra (ex.: CPF em claro) é rejeitada
  R10 client_id deve ser um pseudônimo SHA-256 (64 hex), nunca um identificador cru
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from .config import FEATURES, HOME_OWNERSHIP, ID_COL, LOAN_GRADE, LOAN_INTENT, TARGET, YES_NO


class DataContractError(Exception):
    """Lote violou o contrato e foi bloqueado."""

    def __init__(self, message: str, failure_cases: pd.DataFrame):
        super().__init__(message)
        self.failure_cases = failure_cases


def build_schema(require_target: bool = True) -> pa.DataFrameSchema:
    columns = {
        ID_COL: pa.Column(str, [pa.Check.str_matches(r"^[0-9a-f]{64}$", error="R10_pseudonimo_sha256")],
                          unique=True, nullable=False),
        "person_age": pa.Column(int, pa.Check.in_range(18, 100, error="R2_idade_18_100"), nullable=False),
        "person_income": pa.Column(float, [pa.Check.gt(0, error="R3_renda_positiva"),
                                           pa.Check.le(2_000_000, error="R3_renda_teto")], nullable=False),
        "person_home_ownership": pa.Column(str, pa.Check.isin(HOME_OWNERSHIP, error="R5_moradia"), nullable=False),
        "person_emp_length": pa.Column(float, pa.Check.in_range(0, 60, error="R6_tempo_emprego"), nullable=False),
        "loan_intent": pa.Column(str, pa.Check.isin(LOAN_INTENT, error="R5_finalidade"), nullable=False),
        "loan_grade": pa.Column(str, pa.Check.isin(LOAN_GRADE, error="R5_grade"), nullable=False),
        "loan_amnt": pa.Column(float, pa.Check.in_range(500, 35_000, error="R7_valor_emprestimo"), nullable=False),
        "loan_int_rate": pa.Column(float, pa.Check.in_range(0, 35, error="R7_taxa_juros"), nullable=False),
        "loan_percent_income": pa.Column(float, pa.Check.in_range(0, 1, error="R8_comprometimento"), nullable=False),
        "cb_person_default_on_file": pa.Column(str, pa.Check.isin(YES_NO, error="R5_historico"), nullable=False),
        "cb_person_cred_hist_length": pa.Column(int, pa.Check.in_range(0, 60, error="R6_hist_credito"),
                                                nullable=False),
    }
    if require_target:
        columns[TARGET] = pa.Column(int, pa.Check.isin([0, 1], error="R5_target_binario"), nullable=False)

    return pa.DataFrameSchema(
        columns,
        checks=[
            pa.Check(lambda d: d["person_emp_length"] <= d["person_age"] - 14,
                     element_wise=False, error="R6_emprego_coerente_com_idade"),
            pa.Check(lambda d: (d["loan_percent_income"] - d["loan_amnt"] / d["person_income"]).abs() <= 0.02,
                     element_wise=False, error="R8_coerencia_comprometimento_renda"),
        ],
        unique=FEATURES,      # R4: registro duplicado (mesma pessoa + mesma proposta)
        report_duplicates="all",
        strict=True,          # R9: coluna não prevista (ex.: cpf, nome) => rejeita
        coerce=False,         # tipos errados não são "consertados" silenciosamente
        name="credit_scoring_contract_v1",
    )


def _summarize(fc: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por regra violada, contando registros distintos afetados."""
    fc = fc.copy()
    fc["rule"] = fc["check"].astype(str)
    # checks de DataFrame são reportados uma vez por coluna — colapsa para nível de linha
    fc.loc[fc.schema_context == "DataFrameSchema", "column"] = "<linha>"
    # falhas de schema (coluna extra, dtype) não têm índice de linha: contam como 1 ocorrência
    strict = fc["check"].astype(str) == "column_in_schema"
    fc.loc[strict, "rule"] = "R9_coluna_nao_prevista:" + fc.loc[strict, "failure_case"].astype(str)
    fc.loc[fc["check"].astype(str).str.startswith("dtype"), "rule"] = "tipo_invalido:" + fc["check"].astype(str)
    fc["rule"] = fc["rule"].replace({"not_nullable": "R1_sem_nulos",
                                     "field_uniqueness": "R4_client_id_duplicado",
                                     "multiple_fields_uniqueness": "R4_registro_duplicado"})
    fc["idx"] = fc["index"].astype(object).where(fc["index"].notna(), "schema").astype(str)
    out = (fc.groupby(["schema_context", "column", "rule"], dropna=False)["idx"]
             .nunique().reset_index(name="violations").sort_values("violations", ascending=False))
    return out.reset_index(drop=True)


def validate_batch(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Valida o lote inteiro (lazy) e levanta DataContractError se houver qualquer violação."""
    schema = build_schema(require_target)
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as err:
        fc = err.failure_cases
        summary = _summarize(fc)
        raise DataContractError(
            f"Contrato violado: {int(summary.violations.sum())} falhas em {summary.shape[0]} regras", fc
        ) from None


def summarize_failures(fc: pd.DataFrame) -> pd.DataFrame:
    return _summarize(fc)
