"""Skema request/response API segmentasi. Nama field = `metadata.features` di models/model.json."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE = {
    "customer_id": "C000123",
    "monthly_salary": 11.5,
    "avg_balance": 22.0,
    "investment_balance": 6.5,
    "monthly_txn_count": 70,
    "monthly_txn_amount": 12.0,
    "credit_card_spend": 2.5,
    "loan_outstanding": 4.0,
    "age": 28,
    "tenure_years": 2.0,
    "num_products": 3,
    "digital_txn_ratio": 0.95,
    "preferred_channel": "mobile",
}


class CustomerProfile(BaseModel):
    """Perilaku & nilai nasabah. Semua nominal dalam juta Rupiah."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE})

    customer_id: str | None = Field(None, max_length=64, description="Opsional, hanya dikembalikan di response")
    monthly_salary: float | None = Field(None, ge=0, le=10_000, description="Gaji bulanan; boleh null (diisi median)")
    avg_balance: float = Field(..., ge=0, description="Rata-rata saldo tabungan/giro")
    investment_balance: float = Field(..., ge=0, description="Saldo deposito/reksa dana/obligasi")
    monthly_txn_count: int = Field(..., ge=0, le=100_000)
    monthly_txn_amount: float = Field(..., ge=0)
    credit_card_spend: float = Field(..., ge=0, description="0 jika tidak punya/tidak memakai kartu kredit")
    loan_outstanding: float = Field(..., ge=0)
    age: int = Field(..., ge=17, le=100)
    tenure_years: float = Field(..., ge=0, le=80, description="Lama menjadi nasabah (tahun)")
    num_products: int = Field(..., ge=1, le=30)
    digital_txn_ratio: float = Field(..., ge=0, le=1, description="Porsi transaksi lewat kanal digital")
    preferred_channel: Literal["mobile", "internet_banking", "branch", "atm"]


class BatchRequest(BaseModel):
    instances: list[CustomerProfile] = Field(..., min_length=1, max_length=5000)


class SegmentResult(BaseModel):
    customer_id: str | None
    segment_id: int = Field(..., description="Nomor cluster internal; bisa berubah saat retrain")
    segment_code: str = Field(..., description="Kode persona yang stabil; pakai ini di CRM/campaign")
    segment_name: str
    confidence: float = Field(..., description="GMM: probabilitas posterior; K-Means: skor keanggotaan relatif")
    scores: dict[str, float] = Field(..., description="Skor per segment_code (jumlah = 1)")
    distance_to_center: float
    description: str
    recommendations: list[str]


class BatchResponse(BaseModel):
    model_version: str
    summary: dict[str, int] = Field(..., description="Jumlah nasabah per segment_code")
    results: list[SegmentResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
