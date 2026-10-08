"""FastAPI service untuk segmentasi nasabah (model clustering dalam format JSON)."""

import logging
import os
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import BatchRequest, BatchResponse, CustomerProfile, HealthResponse, SegmentResult
from src.json_model import JsonSegmentModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "model.json"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("segmentation-api")

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    model = JsonSegmentModel.load(MODEL_PATH)
    meta = model.metadata
    input_fields = set(CustomerProfile.model_fields) - {"customer_id"}
    if input_fields != set(meta["features"]):
        raise RuntimeError("Field di schemas.py tidak sama dengan fitur di models/model.json")
    if len(meta["segments"]) != model.n_clusters:
        raise RuntimeError("Jumlah persona di metadata tidak sama dengan jumlah cluster model")
    state["model"], state["metadata"] = model, meta
    state["codes"] = [meta["segments"][str(i)]["code"] for i in range(model.n_clusters)]
    logger.info("Model segmentasi '%s' (%d segmen) versi %s dimuat dari %s",
                meta["model_name"], model.n_clusters, meta["model_version"], MODEL_PATH)
    yield
    state.clear()


app = FastAPI(
    title="Customer Segmentation API",
    description="Menentukan segmen/persona nasabah bank dari perilaku transaksi, saldo, kredit, dan kanal.",
    version="1.0.0",
    lifespan=lifespan,
)


def _segment(rows: list[CustomerProfile]) -> list[SegmentResult]:
    model, meta = state["model"], state["metadata"]
    X = pd.DataFrame([r.model_dump(exclude={"customer_id"}) for r in rows])[meta["features"]]
    X = X.fillna(value=np.nan)
    for col in meta["numeric_features"]:
        X[col] = X[col].astype(float)
    try:
        proba = model.predict_proba(X)
        labels = model.predict(X)
        dist = model.distances(X)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inferensi gagal")
        raise HTTPException(status_code=500, detail="Inferensi model gagal") from exc

    results = []
    for row, lab, p, d in zip(rows, labels, proba, dist):
        seg = meta["segments"][str(int(lab))]
        results.append(SegmentResult(
            customer_id=row.customer_id,
            segment_id=int(lab),
            segment_code=seg["code"],
            segment_name=seg["name"],
            confidence=round(float(p[lab]), 6),
            scores={code: round(float(v), 6) for code, v in zip(state["codes"], p)},
            distance_to_center=round(float(d[lab]), 4),
            description=seg["description"],
            recommendations=seg["recommendations"],
        ))
    return results


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Customer Segmentation API. Buka /docs untuk dokumentasi."}


@app.get("/health", response_model=HealthResponse)
def health():
    meta = state.get("metadata")
    return HealthResponse(status="ok", model_loaded="model" in state, model_version=meta["model_version"] if meta else None)


@app.get("/model-info")
def model_info():
    meta = state["metadata"]
    return {k: meta[k] for k in ("model_name", "model_version", "trained_at", "training_period", "n_training_rows",
                                 "n_clusters", "features", "categorical_features", "evaluation")}


@app.get("/segments")
def segments():
    """Daftar persona: nama, deskripsi, rekomendasi, porsi nasabah, dan profil median."""
    return [{"segment_id": int(i), **s} for i, s in state["metadata"]["segments"].items()]


@app.post("/segment", response_model=SegmentResult)
def segment(payload: CustomerProfile):
    return _segment([payload])[0]


@app.post("/segment/batch", response_model=BatchResponse)
def segment_batch(payload: BatchRequest):
    results = _segment(payload.instances)
    return BatchResponse(model_version=state["metadata"]["model_version"],
                         summary=dict(Counter(r.segment_code for r in results)), results=results)
