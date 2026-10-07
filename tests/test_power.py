import math

import numpy as np
import pandas as pd

from pd_gpnmb.power import (
    backsolve_dispersion,
    condition_se,
    design_matrix,
    fit_nb_wald,
    simulate_power,
    two_sided_normal_p,
)


def _donors(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(50, 95, size=n),
        "sex": np.where(np.arange(n) % 3 == 0, "Female", "Male"),
    })


def test_two_sided_p_matches_reference_values():
    assert math.isclose(two_sided_normal_p(np.array([1.959964]))[0], 0.05, rel_tol=1e-5)
    assert math.isclose(two_sided_normal_p(np.array([0.0]))[0], 1.0)


def test_backsolved_dispersion_reproduces_target_se():
    ctrl, pd_ = _donors(14, 1), _donors(15, 2)
    is_pd = np.r_[np.zeros(14, bool), np.ones(15, bool)]
    ages = np.r_[ctrl["age"], pd_["age"]]
    male = np.r_[ctrl["sex"] == "Male", pd_["sex"] == "Male"]
    X = design_matrix(is_pd, ages, male)
    disp = backsolve_dispersion(X, is_pd, base_mean=170.0, log2fc=1.0, target_lfc_se=0.39)
    mu_ctrl = 170.0 * 29 / (14 + 15 * 2.0)
    se = condition_se(X, np.where(is_pd, 2 * mu_ctrl, mu_ctrl), disp) / math.log(2)
    assert math.isclose(se, 0.39, rel_tol=1e-6)


def test_irls_recovers_known_effect():
    rng = np.random.default_rng(0)
    n = 2000
    cond = np.r_[np.zeros(n), np.ones(n)]
    X = design_matrix(cond, rng.integers(50, 95, 2 * n), rng.integers(0, 2, 2 * n))[None]
    mu = 100 * np.exp(cond * math.log(2.0))
    y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu))[None]
    beta, se = fit_nb_wald(y, X, dispersion=0.5)
    assert abs(beta[0, -1] / math.log(2) - 1.0) < 4 * se[0, -1] / math.log(2)


def test_null_simulation_is_calibrated():
    rng = np.random.default_rng(42)
    out = simulate_power(_donors(14, 1), _donors(15, 2), 15, 0.0, 110.0, 0.5,
                         {"nominal": 0.05}, n_sims=4000, rng=rng)
    assert 0.035 < out["power_nominal"] < 0.07
