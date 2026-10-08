"""Generate dataset sintetis nasabah bank untuk segmentasi -> data/customers.csv.

Data dibangkitkan dari 5 persona tersembunyi dengan distribusi yang saling tumpang
tindih (seperti data nyata). Kolom `_true_persona` HANYA untuk validasi eksternal di
notebook (data nyata tidak punya label ini) dan tidak boleh dipakai model.

Ada juga drift waktu: adopsi digital naik sepanjang 2025 dan porsi nasabah muda-digital
bertambah, untuk menguji stabilitas segmen out-of-time.

    python scripts/generate_data.py --n 12000 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "customers.csv"

# persona: proporsi awal (Jan) -> akhir (Des), lalu parameter perilaku
PERSONAS = {
    "mass_saver": {
        "share": (0.37, 0.31),
        "age": (24, 62), "salary": (4.5, 0.35), "tenure": (6, 4),
        "balance_mult": 1.2, "invest_mult": 0.05, "products": (1, 3),
        "txn_count": 15, "txn_amount_mult": 0.6, "digital": (3, 4),
        "cc_mult": 0.05, "loan_mult": 0.8,
        "occupation": {"factory_worker": 0.35, "retail_service": 0.30, "driver_ojol": 0.20, "private_employee": 0.15},
        "channel": {"atm": 0.45, "branch": 0.30, "mobile": 0.25},
        "city": {"tier1": 0.25, "tier2": 0.40, "tier3": 0.35},
    },
    "digital_young": {
        "share": (0.18, 0.26),
        "age": (22, 35), "salary": (11, 0.30), "tenure": (2, 2),
        "balance_mult": 2.0, "invest_mult": 0.8, "products": (2, 4),
        "txn_count": 65, "txn_amount_mult": 1.0, "digital": (12, 1.5),
        "cc_mult": 0.35, "loan_mult": 0.6,
        "occupation": {"private_employee": 0.55, "professional": 0.30, "soe_employee": 0.15},
        "channel": {"mobile": 0.85, "internet_banking": 0.15},
        "city": {"tier1": 0.65, "tier2": 0.30, "tier3": 0.05},
    },
    "affluent": {
        "share": (0.10, 0.10),
        "age": (40, 66), "salary": (38, 0.35), "tenure": (12, 6),
        "balance_mult": 7.0, "invest_mult": 18.0, "products": (5, 8),
        "txn_count": 32, "txn_amount_mult": 1.4, "digital": (5, 3),
        "cc_mult": 0.40, "loan_mult": 2.5,
        "occupation": {"professional": 0.40, "soe_employee": 0.20, "civil_servant": 0.15, "entrepreneur": 0.25},
        "channel": {"branch": 0.40, "internet_banking": 0.35, "mobile": 0.25},
        "city": {"tier1": 0.75, "tier2": 0.22, "tier3": 0.03},
    },
    "sme_transactor": {
        "share": (0.14, 0.14),
        "age": (28, 56), "salary": (17, 0.50), "tenure": (7, 4),
        "balance_mult": 5.0, "invest_mult": 1.5, "products": (3, 5),
        "txn_count": 140, "txn_amount_mult": 6.0, "digital": (6, 2.5),
        "cc_mult": 0.30, "loan_mult": 6.0,
        "occupation": {"entrepreneur": 0.80, "professional": 0.10, "driver_ojol": 0.10},
        "channel": {"internet_banking": 0.50, "mobile": 0.40, "branch": 0.10},
        "city": {"tier1": 0.45, "tier2": 0.40, "tier3": 0.15},
    },
    "credit_reliant": {
        "share": (0.21, 0.19),
        "age": (27, 52), "salary": (7.5, 0.30), "tenure": (5, 3),
        "balance_mult": 0.5, "invest_mult": 0.02, "products": (3, 5),
        "txn_count": 38, "txn_amount_mult": 0.9, "digital": (5, 3),
        "cc_mult": 0.70, "loan_mult": 14.0,
        "occupation": {"civil_servant": 0.40, "private_employee": 0.35, "factory_worker": 0.15, "retail_service": 0.10},
        "channel": {"mobile": 0.55, "atm": 0.25, "branch": 0.20},
        "city": {"tier1": 0.35, "tier2": 0.45, "tier3": 0.20},
    },
}

SALARIED = {"civil_servant", "soe_employee", "private_employee", "professional", "factory_worker", "retail_service"}


def generate(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    month_idx = rng.integers(0, 12, n)  # snapshot Jan-Des 2025
    t = month_idx / 11

    names = list(PERSONAS)
    start = np.array([PERSONAS[p]["share"][0] for p in names])
    end = np.array([PERSONAS[p]["share"][1] for p in names])
    probs = (1 - t)[:, None] * start + t[:, None] * end
    probs /= probs.sum(axis=1, keepdims=True)
    persona = np.array([rng.choice(names, p=p) for p in probs])

    rows = []
    for i in range(n):
        cfg = PERSONAS[persona[i]]
        occupation = rng.choice(list(cfg["occupation"]), p=list(cfg["occupation"].values()))
        if occupation in SALARIED:
            employee_status = rng.choice(["permanent", "contract", "outsourcing"], p=[0.65, 0.28, 0.07])
            if occupation == "civil_servant":
                employee_status = "permanent"
        else:
            employee_status = "not_applicable"

        age = int(rng.integers(*cfg["age"]))
        salary = cfg["salary"][0] * rng.lognormal(0, cfg["salary"][1])
        tenure = float(np.clip(rng.normal(*cfg["tenure"]), 0.1, age - 17))
        avg_balance = salary * cfg["balance_mult"] * rng.lognormal(0, 0.6)
        investment = salary * cfg["invest_mult"] * rng.lognormal(0, 0.8) * (rng.uniform() < 0.9)
        num_products = int(rng.integers(cfg["products"][0], cfg["products"][1] + 1))
        txn_count = int(max(1, rng.poisson(cfg["txn_count"] * rng.lognormal(0, 0.3))))
        txn_amount = salary * cfg["txn_amount_mult"] * rng.lognormal(0, 0.4)
        digital = float(np.clip(rng.beta(*cfg["digital"]) + 0.12 * t[i], 0, 1))  # adopsi digital naik
        has_cc = rng.uniform() < min(0.95, 0.15 + cfg["cc_mult"] * 1.2)
        cc_spend = salary * cfg["cc_mult"] * rng.lognormal(0, 0.5) if has_cc else 0.0
        loan = salary * cfg["loan_mult"] * rng.lognormal(0, 0.6) * (rng.uniform() < 0.85)

        channel_p = dict(cfg["channel"])
        if "mobile" in channel_p:  # makin banyak yang pindah ke mobile
            shift = 0.15 * t[i] * sum(v for k, v in channel_p.items() if k in ("branch", "atm"))
            for k in ("branch", "atm"):
                if k in channel_p:
                    channel_p[k] *= 1 - 0.15 * t[i]
            channel_p["mobile"] += shift
        channel = rng.choice(list(channel_p), p=np.array(list(channel_p.values())) / sum(channel_p.values()))

        rows.append({
            "customer_id": f"C{i + 1:06d}",
            "snapshot_month": f"2025-{month_idx[i] + 1:02d}",
            "age": age,
            "occupation": occupation,
            "employee_status": employee_status,
            "city_tier": rng.choice(list(cfg["city"]), p=list(cfg["city"].values())),
            "monthly_salary": round(salary, 2),
            "tenure_years": round(tenure, 1),
            "avg_balance": round(avg_balance, 2),
            "investment_balance": round(investment, 2),
            "num_products": num_products,
            "monthly_txn_count": txn_count,
            "monthly_txn_amount": round(txn_amount, 2),
            "digital_txn_ratio": round(digital, 3),
            "credit_card_spend": round(cc_spend, 2),
            "loan_outstanding": round(loan, 2),
            "preferred_channel": channel,
            "_true_persona": persona[i],
        })

    df = pd.DataFrame(rows)
    df.loc[rng.uniform(0, 1, n) < 0.04, "monthly_salary"] = np.nan  # gaji tidak dilaporkan
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = generate(args.n, args.seed)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_PATH, index=False)
    print(f"{data.shape} -> {OUT_PATH}")
    print(data["_true_persona"].value_counts(normalize=True).round(3).to_dict())
