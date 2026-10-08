import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = json.loads((ROOT / "sample_request.json").read_text())
BATCH = json.loads((ROOT / "sample_batch_request.json").read_text())
EXPECTED_CODES = {"affluent", "sme_transactor", "credit_reliant", "digital_young", "mass_saver"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_segments_cover_all_personas(client):
    segs = client.get("/segments").json()
    assert {s["code"] for s in segs} == EXPECTED_CODES
    assert sum(s["share_dev"] for s in segs) == pytest.approx(1.0, abs=1e-6)


def test_model_info_has_evaluation(client):
    body = client.get("/model-info").json()
    assert body["n_clusters"] == 5
    assert body["evaluation"]["bootstrap_ari"]["mean"] > 0.75


def test_segment_single(client):
    body = client.post("/segment", json=SAMPLE).json()
    assert body["customer_id"] == SAMPLE["customer_id"]
    assert body["segment_code"] in EXPECTED_CODES
    assert sum(body["scores"].values()) == pytest.approx(1.0, abs=1e-5)
    assert body["recommendations"]


def test_batch_gives_one_example_per_segment(client):
    # sample_batch_request berisi nasabah paling "khas" untuk setiap segmen
    body = client.post("/segment/batch", json=BATCH).json()
    codes = [r["segment_code"] for r in body["results"]]
    assert set(codes) == EXPECTED_CODES
    assert sum(body["summary"].values()) == len(BATCH["instances"])


def test_null_salary_is_imputed(client):
    assert client.post("/segment", json={**SAMPLE, "monthly_salary": None}).status_code == 200


def test_invalid_channel_rejected(client):
    assert client.post("/segment", json={**SAMPLE, "preferred_channel": "whatsapp"}).status_code == 422


def test_out_of_range_rejected(client):
    assert client.post("/segment", json={**SAMPLE, "digital_txn_ratio": 1.5}).status_code == 422
    assert client.post("/segment", json={**SAMPLE, "avg_balance": -1}).status_code == 422


def test_missing_and_unknown_fields_rejected(client):
    assert client.post("/segment", json={k: v for k, v in SAMPLE.items() if k != "age"}).status_code == 422
    assert client.post("/segment", json={**SAMPLE, "occupation": "entrepreneur"}).status_code == 422
