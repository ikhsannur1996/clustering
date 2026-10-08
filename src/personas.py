"""Penamaan segmen secara otomatis dari profil cluster.

Nomor cluster dari K-Means/GMM acak (cluster 0 hari ini bisa jadi cluster 3 besok), jadi
nama persona TIDAK di-hardcode per nomor. Setiap persona punya aturan berbasis profil
(median fitur per cluster); persona diproses berurutan dan masing-masing mengambil cluster
belum bernama dengan skor tertinggi (greedy). Cluster sisa diberi nama generik.
"""

import numpy as np
import pandas as pd

PERSONAS = [
    {
        "code": "affluent",
        "name": "Nasabah Prioritas (Affluent)",
        "rule": "investment_balance median tertinggi",
        "score": lambda p: p["investment_balance"],
        "description": "Penghasilan & saldo sangat tinggi, investasi besar, memakai banyak produk, nasabah lama.",
        "recommendations": ["Relationship manager / layanan prioritas", "Wealth management: reksa dana, obligasi, bancassurance",
                            "Kartu kredit premium", "Program retensi (risiko churn = kehilangan dana besar)"],
    },
    {
        "code": "sme_transactor",
        "name": "Pelaku Usaha / UMKM",
        "rule": "monthly_txn_count median tertinggi",
        "score": lambda p: p["monthly_txn_count"],
        "description": "Frekuensi & nominal transaksi sangat tinggi (arus kas usaha), saldo menengah-tinggi, pinjaman usaha.",
        "recommendations": ["Rekening giro & cash management bisnis", "Kredit modal kerja / KUR",
                            "QRIS, EDC, payroll untuk karyawan", "Internet banking bisnis multi-user"],
    },
    {
        "code": "credit_reliant",
        "name": "Nasabah Bergantung Kredit",
        "rule": "rasio loan_outstanding / monthly_salary tertinggi",
        "score": lambda p: p["loan_outstanding"] / max(p["monthly_salary"], 0.1),
        "description": "Pinjaman besar relatif terhadap gaji, saldo & investasi sangat rendah, belanja kartu kredit tinggi.",
        "recommendations": ["Pantau risiko kredit (early warning)", "Tawarkan konsolidasi/restrukturisasi utang bila perlu",
                            "Edukasi keuangan & produk tabungan otomatis", "Hindari cross-sell kredit baru yang agresif"],
    },
    {
        "code": "digital_young",
        "name": "Profesional Muda Digital",
        "rule": "digital_txn_ratio tinggi & usia muda (skor = digital_txn_ratio − usia/100)",
        "score": lambda p: p["digital_txn_ratio"] - p["age"] / 100,
        "description": "Usia muda, hampir semua transaksi lewat mobile, gaji menengah, sering bertransaksi, nasabah baru.",
        "recommendations": ["Fitur & promo di aplikasi mobile", "Kartu kredit entry-level / paylater terkontrol",
                            "Investasi reksa dana mulai nominal kecil", "KPR/KKB pertama (life-stage)"],
    },
    {
        "code": "mass_saver",
        "name": "Nasabah Massal Penabung",
        "rule": "monthly_salary median terendah",
        "score": lambda p: -p["monthly_salary"],
        "description": "Gaji di sekitar UMR, saldo kecil, sedikit produk, transaksi jarang, masih banyak memakai ATM/cabang.",
        "recommendations": ["Aktivasi & edukasi mobile banking", "Tabungan berjangka / autodebet nominal kecil",
                            "Asuransi mikro", "Program loyalitas biaya rendah"],
    },
]


def name_segments(profile: pd.DataFrame) -> dict:
    """profile: median fitur per cluster (index = id cluster). Return {cluster_id: persona dict}."""
    remaining = list(profile.index)
    assigned = {}
    for persona in PERSONAS:
        if not remaining:
            break
        scores = {c: persona["score"](profile.loc[c]) for c in remaining}
        best = max(scores, key=scores.get)
        assigned[best] = {k: v for k, v in persona.items() if k != "score"}
        remaining.remove(best)
    for c in remaining:
        assigned[c] = {"code": f"segment_{c}", "name": f"Segmen {c}", "rule": "tidak cocok dengan persona mana pun",
                       "description": "Perlu analisis lanjutan.", "recommendations": []}
    return dict(sorted(assigned.items()))
