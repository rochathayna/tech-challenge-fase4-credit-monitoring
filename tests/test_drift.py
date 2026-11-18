import numpy as np
import pandas as pd

from src.drift_stats import psi_categorical, psi_level, psi_numeric


def test_psi_identical_is_zero():
    x = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert psi_numeric(x, x) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(0, 1, 5000))
    cur = pd.Series(rng.normal(1, 1, 5000))
    assert psi_level(psi_numeric(ref, cur)) == "drift"


def test_psi_same_distribution_is_stable():
    rng = np.random.default_rng(2)
    assert psi_level(psi_numeric(pd.Series(rng.normal(size=5000)), pd.Series(rng.normal(size=5000)))) == "estavel"


def test_psi_categorical_new_category():
    ref = pd.Series(["A"] * 500 + ["B"] * 500)
    cur = pd.Series(["A"] * 300 + ["B"] * 300 + ["C"] * 400)
    assert psi_categorical(ref, cur) > 0.25
