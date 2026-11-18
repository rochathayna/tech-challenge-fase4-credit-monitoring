"""Configuração central do projeto (caminhos, colunas, limiares)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "credit_risk.csv"
PROCESSED_DIR = DATA_DIR / "processed"
PRODUCTION_DIR = DATA_DIR / "production"
QUARANTINE_DIR = DATA_DIR / "quarantine"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"
MLRUNS_DIR = ROOT / "mlruns"

REFERENCE_PATH = PROCESSED_DIR / "reference.csv"
HOLDOUT_PATH = PROCESSED_DIR / "holdout.csv"
MODEL_PATH = MODELS_DIR / "credit_model.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"

RANDOM_STATE = 42

TARGET = "loan_status"          # 1 = inadimplente (default), 0 = adimplente
ID_COL = "client_id"            # pseudônimo (HMAC-SHA256 do CPF)
PREDICTION = "prediction"
PREDICTION_PROBA = "prediction_proba"

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Domínios aceitos pelo contrato de dados
HOME_OWNERSHIP = ["RENT", "OWN", "MORTGAGE", "OTHER"]
LOAN_INTENT = ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
LOAN_GRADE = ["A", "B", "C", "D", "E", "F", "G"]
YES_NO = ["Y", "N"]

# Limiares de monitoramento
PSI_WARN = 0.10       # mudança moderada
PSI_ALERT = 0.25      # mudança significativa
KS_PVALUE = 0.05
DRIFT_SHARE_ALERT = 0.30   # % de features com drift que dispara alerta de dataset
AUC_DROP_ALERT = 0.05      # queda absoluta de ROC AUC vs. baseline
PRODUCTION_MONTHS = 6
