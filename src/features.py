"""Definisi fitur & preprocessing untuk segmentasi nasabah.

Segmentasi dibuat berdasarkan PERILAKU & NILAI nasabah (behavioral segmentation).
Kolom demografis (pekerjaan, status karyawan, kota) dan hasil binning (kelompok umur,
rentang gaji) TIDAK dipakai model, tetapi dipakai untuk MEMBACA profil tiap segmen.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# kolom uang/jumlah yang sangat miring -> log1p dulu agar outlier tidak mendominasi jarak
LOG_COLS = [
    "monthly_salary", "avg_balance", "investment_balance", "monthly_txn_count",
    "monthly_txn_amount", "credit_card_spend", "loan_outstanding",
]
LINEAR_COLS = ["age", "tenure_years", "num_products", "digital_txn_ratio"]
CAT_COLS = ["preferred_channel"]
FEATURES = LOG_COLS + LINEAR_COLS + CAT_COLS

# bobot one-hot: 1 kategori berbeda menambah jarak² sebesar 2*w², setara selisih ~0,7 std
# di satu fitur numerik, jadi kanal ikut berpengaruh tanpa mendominasi 11 fitur numerik
CAT_WEIGHT = 0.5

PROFILE_CAT_COLS = ["occupation", "employee_status", "city_tier"]
ID_COL, DATE_COL, TRUE_LABEL_COL = "customer_id", "snapshot_month", "_true_persona"

# binning untuk profil segmen (bukan input model)
PROFILE_BINS = {
    "age": {"edges": [-np.inf, 25, 35, 45, 55, np.inf], "labels": ["18-25", "26-35", "36-45", "46-55", "56+"]},
    "monthly_salary": {
        "edges": [-np.inf, 3.5, 5, 10, 20, 50, np.inf],
        "labels": ["<3.5jt", "3.5-5jt", "5-10jt", "10-20jt", "20-50jt", ">50jt"],
    },
}


def scale_categories(X, weight=CAT_WEIGHT):
    return np.asarray(X, dtype=float) * weight


def build_preprocessor() -> ColumnTransformer:
    log_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ("weight", FunctionTransformer(scale_categories, feature_names_out="one-to-one")),
    ])
    return ColumnTransformer(
        [("log_num", log_num, LOG_COLS), ("num", num, LINEAR_COLS), ("cat", cat, CAT_COLS)],
        remainder="drop",
    )


def add_profile_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, spec in PROFILE_BINS.items():
        out[f"{col}_band"] = pd.cut(out[col], spec["edges"], labels=spec["labels"])
    return out
