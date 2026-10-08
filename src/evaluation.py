"""Fungsi evaluasi statistik untuk clustering.

Clustering tidak punya label, jadi “cross-validation” dan “backtesting” diganti dengan:
- validasi internal  : silhouette, Calinski-Harabasz, Davies-Bouldin, BIC
- kecenderungan cluster: Hopkins statistic, uji permutasi vs data nol
- stabilitas (≈ CV)  : bootstrap ARI, Jaccard per cluster, split-half agreement
- validasi out-of-time: PSI distribusi segmen, kecocokan dengan model refit, drift centroid
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.base import clone
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


# ============================================================================ validasi internal
def internal_metrics(X, labels, silhouette_sample: int | None = None, seed: int = 42) -> dict:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    return {
        "silhouette": silhouette_score(X, labels, sample_size=silhouette_sample, random_state=seed),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
    }


# ============================================================================ kecenderungan cluster
def hopkins_statistic(X, sample_frac: float = 0.05, seed: int = 42) -> float:
    """Hopkins statistic H (0..1). ≈0,5 = data acak seragam; mendekati 1 = ada struktur cluster.

    Membandingkan jarak tetangga terdekat dari titik acak seragam (u) vs titik data asli (w):
    H = Σu / (Σu + Σw). Batas ruang acak = min–max tiap fitur.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    n, d = X.shape
    m = max(10, int(sample_frac * n))
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    idx = rng.choice(n, m, replace=False)
    w = nn.kneighbors(X[idx], n_neighbors=2)[0][:, 1]
    uniform = rng.uniform(X.min(axis=0), X.max(axis=0), size=(m, d))
    u = nn.kneighbors(uniform, n_neighbors=1)[0][:, 0]
    return float(u.sum() / (u.sum() + w.sum()))


def permute_columns(X, rng) -> np.ndarray:
    """Data nol: setiap kolom diacak sendiri-sendiri. Distribusi marginal sama, struktur bersama hilang."""
    X = np.asarray(X).copy()
    for j in range(X.shape[1]):
        X[:, j] = rng.permutation(X[:, j])
    return X


def permutation_test_silhouette(estimator, X, n_perm: int = 20, sample: int = 4000, seed: int = 42, n_jobs: int = -1) -> dict:
    """Uji permutasi: H0 = struktur cluster tidak lebih kuat daripada data tanpa hubungan antar fitur."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    real_labels = clone(estimator).fit(X).predict(X)
    real = silhouette_score(X, real_labels, sample_size=min(sample, len(X)), random_state=seed)
    seeds = rng.integers(0, 1_000_000, n_perm)

    def one(s):
        r = np.random.default_rng(s)
        Xn = permute_columns(X, r)
        lab = clone(estimator).fit(Xn).predict(Xn)
        return silhouette_score(Xn, lab, sample_size=min(sample, len(X)), random_state=int(s))

    null = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(s) for s in seeds))
    p = (np.sum(null >= real) + 1) / (n_perm + 1)
    return {"silhouette_real": real, "null_mean": null.mean(), "null_std": null.std(ddof=1),
            "null_max": null.max(), "z": (real - null.mean()) / null.std(ddof=1), "p_value": p, "null": null}


# ============================================================================ stabilitas
def _fit_predict(estimator, X_fit, X_pred):
    return clone(estimator).fit(X_fit).predict(X_pred)


def bootstrap_stability(estimator, X, n_boot: int = 20, seed: int = 42, n_jobs: int = -1) -> dict:
    """Stabilitas bootstrap: model dilatih ulang di sampel bootstrap, memprediksi SELURUH data,
    lalu dibandingkan dengan label referensi (ARI) dan per cluster (Jaccard, ala Hennig clusterboot).
    ARI/Jaccard ≥ 0,75 = stabil; 0,6–0,75 = cukup; < 0,6 = tidak stabil.
    """
    X = np.asarray(X)
    rng = np.random.default_rng(seed)
    ref = clone(estimator).fit(X).predict(X)
    idx_list = [rng.choice(len(X), len(X), replace=True) for _ in range(n_boot)]
    preds = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(estimator, X[idx], X) for idx in idx_list)
    aris = np.array([adjusted_rand_score(ref, p) for p in preds])
    clusters = np.unique(ref)
    jaccard = np.zeros((n_boot, len(clusters)))
    for b, p in enumerate(preds):
        for i, c in enumerate(clusters):
            a = ref == c
            jaccard[b, i] = max(np.sum(a & (p == q)) / np.sum(a | (p == q)) for q in np.unique(p))
    return {"ari": aris, "ari_mean": aris.mean(), "ari_std": aris.std(ddof=1),
            "jaccard_per_cluster": pd.DataFrame(jaccard, columns=clusters), "reference_labels": ref}


def split_half_stability(estimator, X, n_repeats: int = 10, seed: int = 42, n_jobs: int = -1) -> np.ndarray:
    """“Cross-validation” untuk clustering: data dibagi 2 bagian acak A & B, model dilatih terpisah
    di A dan di B, lalu keduanya memberi label pada SELURUH data. ARI antar dua pelabelan mengukur
    apakah struktur yang ditemukan bisa direproduksi dari data independen.
    """
    X = np.asarray(X)
    rng = np.random.default_rng(seed)

    def one(s):
        r = np.random.default_rng(s)
        perm = r.permutation(len(X))
        a, b = perm[: len(X) // 2], perm[len(X) // 2:]
        return adjusted_rand_score(_fit_predict(estimator, X[a], X), _fit_predict(estimator, X[b], X))

    return np.array(Parallel(n_jobs=n_jobs)(delayed(one)(s) for s in rng.integers(0, 1_000_000, n_repeats)))


# ============================================================================ pencocokan & drift
def match_clusters(centers_ref, centers_new) -> dict:
    """Hungarian matching: pasangkan cluster model baru ke cluster referensi dengan total jarak minimum."""
    dist = np.linalg.norm(np.asarray(centers_ref)[:, None, :] - np.asarray(centers_new)[None, :, :], axis=2)
    r, c = linear_sum_assignment(dist)
    return {"mapping": dict(zip(c.tolist(), r.tolist())), "distance": dist[r, c]}


def psi_categorical(expected, actual) -> float:
    e = pd.Series(expected).value_counts(normalize=True)
    a = pd.Series(actual).value_counts(normalize=True)
    cats = e.index.union(a.index)
    e = e.reindex(cats, fill_value=1e-6).clip(lower=1e-6)
    a = a.reindex(cats, fill_value=1e-6).clip(lower=1e-6)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(expected, actual, bins: int = 10) -> float:
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def stability_label(value: float) -> str:
    return "stabil" if value < 0.1 else ("perlu perhatian" if value < 0.25 else "bergeser signifikan")


# ============================================================================ uji beda antar segmen
def kruskal_epsilon(values: pd.Series, labels) -> tuple[float, float, float]:
    """Kruskal-Wallis H (beda median > 2 grup, non-parametrik) + epsilon² (besar efek 0..1).
    ε²: < 0,01 sangat kecil, 0,01–0,08 kecil, 0,08–0,26 sedang, ≥ 0,26 besar.
    """
    df = pd.DataFrame({"v": values.to_numpy(), "g": np.asarray(labels)}).dropna()
    groups = [g["v"].to_numpy() for _, g in df.groupby("g")]
    h, p = stats.kruskal(*groups)
    n = len(df)
    return h, p, h / ((n**2 - 1) / (n + 1))


def cramers_v(x, y) -> tuple[float, float, float]:
    table = pd.crosstab(pd.Series(x).astype(str), pd.Series(y))
    chi2, p, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    return chi2, p, float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def holm_correction(p_values: pd.Series) -> pd.Series:
    p = p_values.sort_values()
    m = len(p)
    adj = np.maximum.accumulate([min(1, (m - i) * v) for i, v in enumerate(p.values)])
    return pd.Series(adj, index=p.index).reindex(p_values.index)


def stability_verdict(value: float) -> str:
    return "stabil" if value >= 0.75 else ("cukup" if value >= 0.6 else "tidak stabil")
