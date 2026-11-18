"""Treino do modelo baseline de Credit Scoring (XGBoost) no Dataset de Referência.

O modelo, as métricas e o model card são versionados no MLflow (experimento
`credit-scoring-baseline`) e salvos em models/ para uso no scoring de produção.
"""
from __future__ import annotations

import json
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .config import (CATEGORICAL_FEATURES, FEATURES, MODEL_CARD_PATH, MODEL_PATH, MODELS_DIR,
                     NUMERIC_FEATURES, PREDICTION, PREDICTION_PROBA, PROCESSED_DIR, RANDOM_STATE,
                     REFERENCE_PATH, TARGET)
from .logging_utils import log_event
from .tracking import setup_mlflow

DECISION_THRESHOLD = 0.5
REFERENCE_SCORED_PATH = PROCESSED_DIR / "reference_scored.csv"


def build_model() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])
    clf = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
                        colsample_bytree=0.9, eval_metric="auc", random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


def classification_metrics(y_true, proba, threshold: float = DECISION_THRESHOLD) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "default_rate_real": float(np.mean(y_true)),
        "default_rate_predicted": float(np.mean(pred)),
    }


def score(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[PREDICTION_PROBA] = model.predict_proba(df[FEATURES])[:, 1]
    out[PREDICTION] = (out[PREDICTION_PROBA] >= DECISION_THRESHOLD).astype(int)
    return out


def train() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(REFERENCE_PATH)
    train_df, val_df = train_test_split(ref, test_size=0.25, stratify=ref[TARGET], random_state=RANDOM_STATE)

    setup_mlflow("credit-scoring-baseline")
    with mlflow.start_run(run_name="xgboost-baseline"):
        model = build_model()
        t0 = time.perf_counter()
        model.fit(train_df[FEATURES], train_df[TARGET])
        train_s = time.perf_counter() - t0

        val_scored = score(model, val_df)
        metrics = classification_metrics(val_df[TARGET], val_scored[PREDICTION_PROBA])
        mlflow.log_params({**{k: v for k, v in model.named_steps["clf"].get_params().items()
                              if k in ("n_estimators", "max_depth", "learning_rate", "subsample")},
                           "threshold": DECISION_THRESHOLD, "train_rows": len(train_df),
                           "val_rows": len(val_df), "features": len(FEATURES)})
        mlflow.log_metrics({**metrics, "train_seconds": train_s})

        # importância das features (agregada por coluna original)
        names = model.named_steps["pre"].get_feature_names_out()
        imp = pd.Series(model.named_steps["clf"].feature_importances_, index=names)
        agg = {f: float(imp[[n for n in names if n.split("__", 1)[1].startswith(f)]].sum()) for f in FEATURES}
        importance = dict(sorted(agg.items(), key=lambda kv: -kv[1]))

        joblib.dump(model, MODEL_PATH)
        val_scored.to_csv(REFERENCE_SCORED_PATH, index=False)
        card = {
            "model": "XGBClassifier (sklearn Pipeline + OneHotEncoder)",
            "version": "1.0.0",
            "objective": "Probabilidade de inadimplência (loan_status=1) para apoio à decisão de crédito",
            "decision_threshold": DECISION_THRESHOLD,
            "training_data": {"rows": len(train_df), "default_rate": float(train_df[TARGET].mean())},
            "validation_metrics": metrics,
            "feature_importance": importance,
            "features": FEATURES,
            "excluded_attributes": ["cpf (pseudonimizado e descartado)", "sexo", "raça/cor", "religião",
                                    "orientação sexual", "dados de saúde"],
            "intended_use": "Apoio à decisão com revisão humana garantida (art. 20 LGPD)",
        }
        MODEL_CARD_PATH.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(MODEL_CARD_PATH))
        mlflow.sklearn.log_model(model, name="model", input_example=train_df[FEATURES].head(3),
                                 skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"])

    log_event("training", "model_trained", "success", roc_auc=round(metrics["roc_auc"], 4),
              f1=round(metrics["f1"], 4), train_rows=len(train_df), train_seconds=round(train_s, 2))
    return {"metrics": metrics, "importance": importance}
