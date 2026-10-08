import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from src.evaluation import (
    bootstrap_stability,
    hopkins_statistic,
    kruskal_epsilon,
    match_clusters,
    permutation_test_silhouette,
    psi_categorical,
    psi_numeric,
    split_half_stability,
)
from src.personas import name_segments

X_BLOBS, Y_BLOBS = make_blobs(n_samples=1500, centers=4, cluster_std=0.6, random_state=0)
X_UNIFORM = np.random.default_rng(0).uniform(-5, 5, size=(1500, 2))


def test_hopkins_separates_clustered_from_uniform():
    assert hopkins_statistic(X_BLOBS) > 0.85
    assert 0.4 < hopkins_statistic(X_UNIFORM) < 0.6


def test_bootstrap_and_split_half_stable_on_clear_blobs():
    est = KMeans(4, n_init=5, random_state=0)
    assert bootstrap_stability(est, X_BLOBS, n_boot=5)["ari_mean"] > 0.95
    assert split_half_stability(est, X_BLOBS, n_repeats=5).mean() > 0.95


def test_permutation_test_significant_for_blobs():
    res = permutation_test_silhouette(KMeans(4, n_init=5, random_state=0), X_BLOBS, n_perm=9, n_jobs=1)
    assert res["p_value"] == pytest.approx(0.1)  # minimum mungkin dengan 9 permutasi
    assert res["silhouette_real"] > res["null_max"]


def test_match_clusters_recovers_permutation():
    centers = np.array([[0, 0], [5, 5], [10, 0]], dtype=float)
    shuffled = centers[[2, 0, 1]] + 0.01
    m = match_clusters(centers, shuffled)
    assert m["mapping"] == {0: 2, 1: 0, 2: 1}


def test_psi_identical_zero():
    x = np.random.default_rng(1).normal(size=3000)
    assert psi_numeric(x, x) == pytest.approx(0, abs=1e-9)
    assert psi_categorical(["a", "b"] * 100, ["a", "b"] * 100) == pytest.approx(0, abs=1e-9)


def test_kruskal_effect_size_large_for_separated_groups():
    _, p, eps = kruskal_epsilon(pd.Series(X_BLOBS[:, 0]), Y_BLOBS)
    assert p < 1e-10 and eps > 0.26


def test_persona_naming_is_independent_of_cluster_numbering():
    profile = pd.DataFrame({
        "investment_balance": [600, 20, 0.1, 7, 0.2],
        "monthly_txn_count": [30, 140, 38, 64, 15],
        "loan_outstanding": [75, 90, 99, 5, 3],
        "monthly_salary": [38, 17, 7.6, 11, 4.5],
        "digital_txn_ratio": [0.67, 0.76, 0.68, 0.94, 0.47],
        "age": [53, 42, 39, 28, 42],
    })
    expected = ["affluent", "sme_transactor", "credit_reliant", "digital_young", "mass_saver"]
    assert [s["code"] for s in name_segments(profile).values()] == expected
    shuffled = profile.iloc[[3, 0, 4, 1, 2]].reset_index(drop=True)
    assert [s["code"] for s in name_segments(shuffled).values()] == [expected[i] for i in [3, 0, 4, 1, 2]]


def test_games_howell_detects_only_real_difference():
    from src.evaluation import games_howell
    rng = np.random.default_rng(1)
    v = np.concatenate([rng.normal(0, 1, 300), rng.normal(0, 3, 300), rng.normal(1, 1, 300)])
    g = np.repeat(["a", "b", "c"], 300)
    res = games_howell(v, g).set_index(["grup_a", "grup_b"])
    assert res.loc[("a", "b"), "p_value"] > 0.05      # rata-rata sama, varians beda
    assert res.loc[("a", "c"), "p_value"] < 1e-6
