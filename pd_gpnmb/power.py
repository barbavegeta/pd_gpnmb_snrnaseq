from __future__ import annotations

import math

import numpy as np
import pandas as pd


def two_sided_normal_p(z: np.ndarray) -> np.ndarray:
    """Two-sided p-value for a Wald z statistic (numpy-only, no scipy)."""
    erfc = np.frompyfunc(math.erfc, 1, 1)
    return erfc(np.abs(np.asarray(z, dtype=float)) / math.sqrt(2.0)).astype(float)


def design_matrix(condition_pd: np.ndarray, age: np.ndarray, sex_male: np.ndarray) -> np.ndarray:
    """Intercept + centred age + sex + condition, matching `~ age + sex + condition`.

    Age is centred for numerical stability; this does not change the condition coefficient.
    Works on 1-D inputs (one design) or 2-D inputs (one design per simulation).
    """
    age = np.asarray(age, dtype=float)
    age_c = age - age.mean(axis=-1, keepdims=True)
    cols = [np.ones_like(age_c), age_c, np.asarray(sex_male, dtype=float), np.asarray(condition_pd, dtype=float)]
    return np.stack(cols, axis=-1)


def condition_se(X: np.ndarray, mu: np.ndarray, dispersion: float) -> float:
    """Wald standard error (natural-log scale) of the last coefficient for an NB GLM."""
    w = mu / (1.0 + dispersion * mu)
    info = X.T @ (w[:, None] * X)
    return float(math.sqrt(np.linalg.inv(info)[-1, -1]))


def backsolve_dispersion(
    X: np.ndarray,
    condition_pd: np.ndarray,
    base_mean: float,
    log2fc: float,
    target_lfc_se: float,
) -> float:
    """Find the NB dispersion that reproduces an observed DESeq2 log2FC standard error.

    Group means are set so that their sample-weighted average equals DESeq2's baseMean and
    their ratio equals the observed fold change. The returned dispersion is therefore the
    one implied by the real design, which absorbs DESeq2's dispersion shrinkage.
    """
    condition_pd = np.asarray(condition_pd, dtype=bool)
    ratio = 2.0 ** log2fc
    n_pd, n_ctrl = condition_pd.sum(), (~condition_pd).sum()
    mu_ctrl = base_mean * (n_pd + n_ctrl) / (n_ctrl + n_pd * ratio)
    mu = np.where(condition_pd, mu_ctrl * ratio, mu_ctrl)
    target = target_lfc_se * math.log(2.0)

    lo, hi = 1e-8, 50.0
    if condition_se(X, mu, lo) > target:
        raise ValueError("Observed SE is smaller than the Poisson limit; cannot back-solve dispersion.")
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if condition_se(X, mu, mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def fit_nb_wald(y: np.ndarray, X: np.ndarray, dispersion: float, n_iter: int = 50, tol: float = 1e-10):
    """Batched negative binomial GLM (log link, fixed dispersion) fitted by IRLS.

    y: (S, n) counts; X: (S, n, p) designs. Returns (beta, se) for every coefficient,
    on the natural-log scale, each shaped (S, p).
    """
    y = np.asarray(y, dtype=float)
    S, n, p = X.shape
    beta = np.zeros((S, p))
    beta[:, 0] = np.log(np.maximum(y.mean(axis=1), 0.5))
    for _ in range(n_iter):
        eta = np.einsum("snp,sp->sn", X, beta)
        mu = np.exp(eta)
        w = mu / (1.0 + dispersion * mu)
        z = eta + (y - mu) / mu
        xtwx = np.einsum("snp,sn,snq->spq", X, w, X)
        xtwz = np.einsum("snp,sn,sn->sp", X, w, z)
        new = np.linalg.solve(xtwx, xtwz[..., None])[..., 0]
        step = np.max(np.abs(new - beta))
        beta = new
        if step < tol:
            break
    mu = np.exp(np.einsum("snp,sp->sn", X, beta))
    w = mu / (1.0 + dispersion * mu)
    cov = np.linalg.inv(np.einsum("snp,sn,snq->spq", X, w, X))
    se = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    return beta, se


def simulate_power(
    donors_ctrl: pd.DataFrame,
    donors_pd: pd.DataFrame,
    n_per_group: int,
    log2fc: float,
    mu_ctrl: float,
    dispersion: float,
    thresholds: dict[str, float],
    n_sims: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Simulate donor-level pseudobulk counts for one gene and test PD vs control.

    Age and sex are resampled from the real donors within each condition, so the
    simulated cohorts keep the observed covariate structure as the cohort grows.
    """
    def draw(pool: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.integers(0, len(pool), size=(n_sims, n_per_group))
        return pool["age"].to_numpy(float)[idx], (pool["sex"].to_numpy() == "Male")[idx].astype(float)

    age_c, sex_c = draw(donors_ctrl)
    age_p, sex_p = draw(donors_pd)
    age = np.concatenate([age_c, age_p], axis=1)
    sex = np.concatenate([sex_c, sex_p], axis=1)
    cond = np.concatenate([np.zeros((n_sims, n_per_group)), np.ones((n_sims, n_per_group))], axis=1)
    X = design_matrix(cond, age, sex)
    # Small resampled cohorts can be all one sex (or sex fully confounded with condition);
    # redraw those, as the real analysis would never fit a rank-deficient design.
    bad = np.linalg.matrix_rank(X) < X.shape[-1]
    while bad.any():
        k = int(bad.sum())
        a_c, s_c = draw(donors_ctrl)
        a_p, s_p = draw(donors_pd)
        age[bad] = np.concatenate([a_c[:k], a_p[:k]], axis=1)
        sex[bad] = np.concatenate([s_c[:k], s_p[:k]], axis=1)
        X[bad] = design_matrix(cond[bad], age[bad], sex[bad])
        bad = np.linalg.matrix_rank(X) < X.shape[-1]

    mu = mu_ctrl * np.exp(cond * log2fc * math.log(2.0))
    size = 1.0 / dispersion
    y = rng.negative_binomial(size, size / (size + mu))

    beta, se = fit_nb_wald(y, X, dispersion)
    lfc_hat = beta[:, -1] / math.log(2.0)
    p = two_sided_normal_p(beta[:, -1] / se[:, -1])
    out = {
        "n_per_group": n_per_group,
        "true_log2fc": log2fc,
        "mean_log2fc_hat": float(lfc_hat.mean()),
        "median_lfc_se": float(np.median(se[:, -1]) / math.log(2.0)),
    }
    for name, alpha in thresholds.items():
        out[f"power_{name}"] = float(np.mean(p <= alpha))
    return out
