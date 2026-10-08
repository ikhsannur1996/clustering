# Segmentasi Nasabah Bank (Customer Segmentation): End-to-End Clustering

Proyek clustering perbankan dari notebook sampai API: mengelompokkan nasabah berdasarkan **perilaku transaksi, saldo, investasi, kredit, dan kanal**, lalu memberi setiap segmen **persona** dan **rekomendasi produk**. Dua model dibandingkan (**K-Means** vs **Gaussian Mixture**), model terbaik disimpan sebagai **JSON**, disajikan lewat **FastAPI**, dan dijalankan di **Docker lokal**.

**Highlight:**
- Dataset 12.000 nasabah (snapshot Jan–Des 2025): 12 fitur perilaku untuk model + pekerjaan, status karyawan, dan tier kota untuk profil
- Preprocessing di pipeline: **log1p** untuk kolom uang yang miring, **scaling**, **one-hot berbobot** untuk kanal; **binning** umur & gaji (rentang UMR) untuk membaca profil
- Pemilihan k dengan aturan tertulis: rentang bisnis 4–8, **stabilitas bootstrap ≥ 0,75**, silhouette tertinggi
- Evaluasi statistik lengkap: Hopkins, **uji permutasi**, **bootstrap stability** (ARI + Jaccard per cluster), **split-half cross-validation**, Kruskal-Wallis/chi-square antar segmen
- **Validasi out-of-time** (padanan backtesting): PSI distribusi segmen, refit + pencocokan Hungarian, drift centroid, CSI fitur
- Persona dinamai **otomatis** berdasarkan aturan profil, sehingga kode segmen tetap konsisten walaupun nomor cluster berubah saat retrain
- Model JSON ~11 KB, sehingga API **tidak butuh scikit-learn**

## Hasil

**Model terpilih: K-Means, k = 5** (menang 6 dari 6 kriteria internal vs Gaussian Mixture).

| Segmen (`segment_code`) | Porsi | Ciri utama | % dana | % investasi | % kredit | % transaksi |
|---|---|---|---|---|---|---|
| Nasabah Massal Penabung (`mass_saver`) | 35% | gaji ±UMR, saldo kecil, masih ATM/cabang | 4 | 0,1 | 3 | 4 |
| Profesional Muda Digital (`digital_young`) | 22% | usia ±28, 94% transaksi digital, nasabah baru | 10 | 2,5 | 3 | 10 |
| Nasabah Bergantung Kredit (`credit_reliant`) | 19% | pinjaman ±13× gaji, saldo sangat kecil | 1,5 | 0 | **44** | 5 |
| Pelaku Usaha / UMKM (`sme_transactor`) | 14% | 137 transaksi/bulan, nominal besar, pinjaman usaha | 29 | 5 | 33 | **60** |
| Nasabah Prioritas (`affluent`) | 10% | gaji 38 jt, investasi 598 jt, 6 produk | **55** | **92** | 18 | 21 |

### Ringkasan Evaluasi Statistik (15 ✅ · 1 ⚠️ · 0 ❌)

| Aspek | Hasil | Status |
|---|---|---|
| Struktur cluster nyata | Hopkins 0,789 (data nol 0,658); uji permutasi silhouette 0,31 vs 0,08, z = 104, p = 0,01 | ✅ |
| Kualitas internal | silhouette 0,312; hanya 1,5% nasabah silhouette < 0; segmen terkecil 10% | ✅ |
| Stabilitas (≈ cross-validation) | bootstrap ARI 0,996 ± 0,002; Jaccard cluster terlemah 0,994; split-half ARI 0,993 | ✅ |
| Makna segmen | pekerjaan (tidak dipakai model) berbeda tegas antar segmen: Cramér's V 0,63 | ✅ |
| Out-of-time (Okt–Des 2025) | silhouette 0,318 (dev 0,312); refit OOT ARI 0,981 (99,2% label sama); PSI segmen 0,015 | ✅ |
| Drift fitur | `digital_txn_ratio` CSI 0,148, karena adopsi digital naik di semua segmen | ⚠️ pantau / retrain berkala |
| Validasi eksternal* | ARI vs persona asli: K-Means 0,944, GMM 0,997 (*hanya mungkin di data sintetis) | ✅ |

> **Catatan jujur:** metrik internal memilih K-Means karena lebih kompak dan lebih stabil, padahal GMM sedikit lebih mirip persona asli. Di data nyata kebenaran tidak diketahui, jadi keputusan mengikuti aturan yang ditetapkan sebelum melihat hasil. Penjelasan lengkap ada di notebook Bagian 9 & 12.

## Struktur Proyek

```
clustering-e2e/
├── data/customers.csv                    # 12.000 nasabah (dibuat oleh scripts/generate_data.py)
├── scripts/generate_data.py              # generator data sintetis: 5 persona + drift digital sepanjang 2025
├── notebooks/customer_segmentation_end_to_end.ipynb
├── src/
│   ├── features.py                       # daftar fitur, preprocessing (log1p, scaling, one-hot berbobot), binning profil
│   ├── evaluation.py                     # Hopkins, uji permutasi, bootstrap & split-half stability, PSI, Kruskal-Wallis, dll.
│   ├── personas.py                       # penamaan persona otomatis + deskripsi & rekomendasi produk
│   └── json_model.py                     # ekspor pipeline → JSON + inferensi (numpy/pandas)
├── models/model.json                     # preprocessing + centroid + persona + ringkasan evaluasi
├── app/
│   ├── main.py                           # FastAPI
│   └── schemas.py                        # validasi input (Pydantic)
├── tests/
│   ├── test_api.py                       # endpoint & validasi input
│   └── test_evaluation.py                # fungsi statistik & penamaan persona
├── sample_request.json / sample_batch_request.json
├── requirements.txt                      # API (tanpa scikit-learn)
├── requirements-dev.txt                  # + scikit-learn, scipy, notebook, testing
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
```

---

## 1. Data & Notebook

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# (opsional) buat ulang dataset
python scripts/generate_data.py --n 12000 --seed 42

jupyter lab notebooks/customer_segmentation_end_to_end.ipynb
```

Jalankan semua cell (*Run All*, ±1 menit). Hasilnya menimpa `models/model.json` dan `sample_*.json`.

Isi notebook (18 bagian; setiap cell kode didahului kotak **Alur data: Input → Proses → Output → Berikutnya**):

| Bagian | Isi |
|---|---|
| 0–3 | Peta alur data, import, load & kamus data, EDA (missing, nilai nol, skewness & efek log1p, korelasi) |
| 4 | Split **waktu**: dev Jan–Sep 2025, out-of-time Okt–Des 2025 |
| 5 | Preprocessing (log1p, scaling, one-hot × 0,5) + penelusuran 1 nasabah |
| 6 | **Uji kecenderungan cluster**: Hopkins vs data nol, peta kepadatan PCA |
| 7 | 2 model kandidat + demonstrasi kenapa GMM butuh `reg_covar` |
| 8 | **Pencarian k** (2–10): inertia, BIC/AIC, silhouette, CH, DB, stabilitas bootstrap, aturan pemilihan k |
| 9 | **Evaluasi mendalam** (padanan cross-validation): diagram silhouette, bootstrap 50×, split-half 20×, uji permutasi 99×, pemilihan model |
| 10 | Model final + **penamaan persona otomatis** |
| 11 | **Profil segmen**: indeks relatif, PCA, binning umur & gaji, demografi, Kruskal-Wallis + ε², chi-square + Cramér's V |
| 12 | **Validasi eksternal** vs persona asli (khusus data sintetis) |
| 13 | **Validasi out-of-time**: komposisi segmen (χ², PSI, CI Wilson), silhouette OOT, refit + Hungarian, drift centroid, CSI |
| 14 | Nilai bisnis per segmen, rekomendasi, **ringkasan evaluasi** (16 pengujian), kesimpulan |
| 15–17 | Ekspor JSON, validasi identik dengan scikit-learn, alur data di API |

## 2. Menjalankan API Lokal (tanpa Docker)

```bash
uvicorn app.main:app --reload --port 8001
```

Buka **http://localhost:8001/docs**. Untuk test:

```bash
pytest -q
```

## 3. Deploy ke Docker Lokal

### Prasyarat
- Docker Desktop terpasang dan **berjalan** (cek: `docker info`)
- `models/model.json` sudah ada (hasil notebook)

### Opsi A: Docker CLI

```bash
docker build -t customer-segmentation-api:latest .
docker run -d --name customer-segmentation-api -p 8001:8000 customer-segmentation-api:latest
docker ps                                   # tunggu STATUS = healthy
docker logs -f customer-segmentation-api
```

### Opsi B: Docker Compose

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f api
```

> Port host **8001** dipakai agar bisa berjalan bersamaan dengan API credit default (port 8000).

### Uji API

```bash
curl http://localhost:8001/health
curl http://localhost:8001/segments          # daftar persona + profil + rekomendasi
curl http://localhost:8001/model-info        # termasuk ringkasan evaluasi statistik

# 1 nasabah
curl -X POST http://localhost:8001/segment \
  -H "Content-Type: application/json" \
  -d @sample_request.json

# batch (maks. 5.000), sekaligus ringkasan jumlah per segmen
curl -X POST http://localhost:8001/segment/batch \
  -H "Content-Type: application/json" \
  -d @sample_batch_request.json
```

Contoh request (semua nominal dalam **juta Rp**):

```json
{
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
  "preferred_channel": "mobile"
}
```

Response:

```json
{
  "customer_id": "C000123",
  "segment_id": 2,
  "segment_code": "digital_young",
  "segment_name": "Profesional Muda Digital",
  "confidence": 0.999829,
  "scores": {"sme_transactor": 2.4e-05, "mass_saver": 0.0, "digital_young": 0.999829, "credit_reliant": 0.000147, "affluent": 0.0},
  "distance_to_center": 0.3593,
  "description": "Usia muda, hampir semua transaksi lewat mobile, gaji menengah, sering bertransaksi, nasabah baru.",
  "recommendations": ["Fitur & promo di aplikasi mobile", "Kartu kredit entry-level / paylater terkontrol",
                      "Investasi reksa dana mulai nominal kecil", "KPR/KKB pertama (life-stage)"]
}
```

- **`segment_code`** stabil antar retrain. Pakai ini di CRM/campaign, **bukan** `segment_id`.
- **`confidence` / `scores`**: K-Means tidak punya probabilitas, jadi nilainya adalah skor keanggotaan relatif softmax(−jarak²), berguna untuk mendeteksi nasabah di perbatasan dua segmen. Jika model terpilih GMM, nilainya adalah probabilitas posterior.
- **`distance_to_center`**: jarak ke pusat segmen (satuan std). Nilai yang jauh di atas median segmen (lihat `/segments` dan notebook Bagian 13) menandakan nasabah atipikal.

### Stop & Bersihkan

```bash
docker stop customer-segmentation-api && docker rm customer-segmentation-api   # Opsi A
docker compose down                                                             # Opsi B
docker rmi customer-segmentation-api:latest                                     # opsional
```

### Update Model (Retrain)

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/customer_segmentation_end_to_end.ipynb
docker compose up -d --build
curl http://localhost:8001/health      # model_version berubah
```

Atau tanpa rebuild image: `-v "$(pwd)/models:/app/models:ro"` lalu `docker restart customer-segmentation-api`.

---

## Referensi API

| Method | Endpoint | Keterangan |
|---|---|---|
| GET | `/health` | Status & versi model |
| GET | `/model-info` | Info model, periode training, fitur, ringkasan evaluasi |
| GET | `/segments` | Persona: nama, deskripsi, rekomendasi, porsi dev/OOT, median profil |
| POST | `/segment` | Segmen 1 nasabah |
| POST | `/segment/batch` | `{"instances": [...]}` → hasil per nasabah + jumlah per segmen |
| GET | `/docs` | Swagger UI |

**Validasi input** (HTTP 422 jika gagal): 11 field numerik wajib (kecuali `monthly_salary` boleh `null`), semua nominal ≥ 0, `digital_txn_ratio` 0–1, `preferred_channel` ∈ {mobile, internet_banking, branch, atm}, dan field yang tidak dikenal ditolak. Pekerjaan, status karyawan, dan kota **tidak** diperlukan karena tidak dipakai model.

| Env var | Default | Keterangan |
|---|---|---|
| `MODEL_PATH` | `models/model.json` | Lokasi model JSON |
| `LOG_LEVEL` | `INFO` | Level logging |

## Monitoring yang Disarankan

| Indikator | Patokan | Tindakan |
|---|---|---|
| PSI distribusi segmen (bulanan) | < 0,1 stabil; 0,1–0,25 investigasi; > 0,25 | retrain |
| CSI per fitur | sama | cari penyebab (mis. kampanye digital, produk baru) |
| Silhouette pada data baru | turun > 0,03 dari dev | retrain |
| ARI model lama vs refit data baru | < 0,75 | segmen berubah struktur → tinjau ulang persona |
| Jadwal | setiap 6 bulan | retrain + cek ulang pemetaan persona |

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `Cannot connect to the Docker daemon` | Buka Docker Desktop dan tunggu sampai *running* |
| `port is already allocated` | Ganti port host, mis. `-p 8002:8000` |
| `ModuleNotFoundError: No module named 'src'` | Jalankan uvicorn dari root proyek; di Docker pastikan `src/json_model.py` ikut di-COPY |
| `Jumlah persona di metadata tidak sama…` saat start | `model.json` rusak/tidak lengkap. Jalankan ulang notebook |
| Response 422 | Cek nama field, nilai `preferred_channel`, dan rentang angka di `/docs` |
| Nama segmen berubah setelah retrain | Normal untuk `segment_id`. `segment_code` tetap konsisten karena persona dinamai berdasarkan aturan profil |

> ⚠️ Dataset bersifat sintetis untuk pembelajaran. Di data nyata, segmen biasanya lebih tumpang tindih (silhouette lebih rendah), tetapi prosedur evaluasinya tetap sama.
