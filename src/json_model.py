"""Serialisasi pipeline segmentasi ke JSON + inferensi tanpa scikit-learn.

Mendukung preprocessing dari `src/features.py` dan model KMeans / GaussianMixture
(covariance full atau diag). Inferensi hanya memakai numpy & pandas.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FORMAT_NAME = "customer-segmentation-json-model"
FORMAT_VERSION = 1


def _to_list(a):
    return np.asarray(a).tolist()


# ----------------------------------------------------------------------------- export
def _export_preprocessor(ct) -> list[dict]:
    steps = []
    for name, tr, cols in ct.transformers_:
        if name == "remainder":
            continue
        if name in ("log_num", "num"):
            steps.append({
                "name": name, "type": "impute_log_scale" if name == "log_num" else "impute_scale",
                "columns": list(cols),
                "impute_values": _to_list(tr.named_steps["imputer"].statistics_),
                "log1p": name == "log_num",
                "mean": _to_list(tr.named_steps["scaler"].mean_),
                "scale": _to_list(tr.named_steps["scaler"].scale_),
            })
        elif name == "cat":
            weight_step = tr.named_steps["weight"]
            steps.append({
                "name": name, "type": "impute_onehot_weight", "columns": list(cols),
                "impute_values": _to_list(tr.named_steps["imputer"].statistics_),
                "categories": [_to_list(c) for c in tr.named_steps["onehot"].categories_],
                "weight": float((weight_step.kw_args or {}).get("weight", weight_step.func.__defaults__[0])),
            })
        else:
            raise ValueError(f"Transformer '{name}' belum didukung")
    return steps


def _export_model(model) -> dict:
    kind = type(model).__name__
    if kind == "KMeans":
        return {"type": "kmeans", "centers": _to_list(model.cluster_centers_)}
    if kind == "GaussianMixture":
        if model.covariance_type not in ("full", "diag"):
            raise ValueError("Hanya covariance_type full/diag yang didukung")
        return {"type": "gmm", "covariance_type": model.covariance_type, "weights": _to_list(model.weights_),
                "means": _to_list(model.means_), "precisions_cholesky": _to_list(model.precisions_cholesky_)}
    raise ValueError(f"Model '{kind}' belum didukung")


def export_pipeline(pipeline, metadata: dict) -> dict:
    ct = pipeline.named_steps["preprocess"]
    return {"format": FORMAT_NAME, "format_version": FORMAT_VERSION, "metadata": metadata,
            "preprocessing": _export_preprocessor(ct), "feature_names_out": _to_list(ct.get_feature_names_out()),
            "model": _export_model(pipeline.named_steps["model"])}


def save_json(spec: dict, path) -> None:
    Path(path).write_text(json.dumps(spec, indent=1, allow_nan=False, ensure_ascii=False))


# ----------------------------------------------------------------------------- inference
class JsonSegmentModel:
    def __init__(self, spec: dict):
        if spec.get("format") != FORMAT_NAME or spec.get("format_version") != FORMAT_VERSION:
            raise ValueError("Format model JSON tidak dikenali")
        self.spec, self.metadata, self.model = spec, spec["metadata"], spec["model"]
        self.features = self.metadata["features"]
        m = self.model
        if m["type"] == "kmeans":
            self.centers = np.asarray(m["centers"])
        else:
            self.weights, self.means = np.asarray(m["weights"]), np.asarray(m["means"])
            self.prec_chol = np.asarray(m["precisions_cholesky"])

    @classmethod
    def load(cls, path) -> "JsonSegmentModel":
        return cls(json.loads(Path(path).read_text()))

    @property
    def n_clusters(self) -> int:
        return len(self.centers) if self.model["type"] == "kmeans" else len(self.weights)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        blocks = []
        for step in self.spec["preprocessing"]:
            cols = step["columns"]
            if step["type"] in ("impute_log_scale", "impute_scale"):
                x = X[cols].astype(float).to_numpy()
                x = np.where(np.isnan(x), np.asarray(step["impute_values"], dtype=float), x)
                if step["log1p"]:
                    x = np.log1p(x)
                blocks.append((x - np.asarray(step["mean"])) / np.asarray(step["scale"]))
            elif step["type"] == "impute_onehot_weight":
                for col, fill, cats in zip(cols, step["impute_values"], step["categories"]):
                    v = X[col].astype(object).where(X[col].notna(), fill).to_numpy()
                    blocks.append((v[:, None] == np.asarray(cats, dtype=object)[None, :]).astype(float) * step["weight"])
            else:
                raise ValueError(step["type"])
        out = np.hstack(blocks)
        assert out.shape[1] == len(self.spec["feature_names_out"])
        return out

    def _gmm_log_resp(self, Z):
        n, d = Z.shape
        log_prob = np.empty((n, len(self.weights)))
        for k in range(len(self.weights)):
            pc = self.prec_chol[k]
            if self.model["covariance_type"] == "full":
                y = Z @ pc - self.means[k] @ pc
                log_det = np.sum(np.log(np.diag(pc)))
            else:
                y = (Z - self.means[k]) * pc
                log_det = np.sum(np.log(pc))
            log_prob[:, k] = -0.5 * (d * np.log(2 * np.pi) + np.sum(y**2, axis=1)) + log_det
        weighted = log_prob + np.log(self.weights)
        mx = weighted.max(axis=1, keepdims=True)
        return weighted - (mx + np.log(np.exp(weighted - mx).sum(axis=1, keepdims=True)))

    def distances(self, X: pd.DataFrame) -> np.ndarray:
        """Jarak Euclidean ke pusat setiap segmen (centroid K-Means atau mean komponen GMM)."""
        Z = self.transform(X[self.features])
        centers = self.centers if self.model["type"] == "kmeans" else self.means
        return np.linalg.norm(Z[:, None, :] - centers[None, :, :], axis=2)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """GMM: probabilitas posterior. K-Means: tidak punya probabilitas, jadi dipakai skor
        keanggotaan lunak softmax(−jarak²) yang hanya untuk ranking/keyakinan relatif."""
        if self.model["type"] == "gmm":
            return np.exp(self._gmm_log_resp(self.transform(X[self.features])))
        d2 = self.distances(X) ** 2
        z = -(d2 - d2.min(axis=1, keepdims=True))
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model["type"] == "kmeans":
            return self.distances(X).argmin(axis=1)
        return self._gmm_log_resp(self.transform(X[self.features])).argmax(axis=1)
