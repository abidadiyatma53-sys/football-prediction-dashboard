# ⚽ Football Data Ingestion, Quantitative Modeling & Prediction Pipeline

Sistem end-to-end data engineering, model kuantitatif sepak bola (Dixon-Coles Poisson & XGBoost Multi-Output Regression), Value Betting Engine (+EV & Kelly Criterion), audit trail keuangan (SQLite Ledger), dan multi-container production deployment (FastAPI & Streamlit).

---

## 🏛️ Arsitektur Sistem Produksi

```
football-data-pipeline/
│
├── config.py                 # Konfigurasi liga, season mapping, HTTP headers & retry policy
├── normalizer.py             # Normalisasi nama klub kanonikal lintas sumber data
├── cleaner.py                # Pembersihan data (deduplikasi, xG alignment, type casting)
├── pipeline.py               # Orchestrator scraping ETL utama
├── features.py               # Feature Engineering (xG, Elo, Rest Days, Venue Form, Attack/Defense)
├── elo.py                    # World Football Dynamic Elo Rating Engine (FIFA/FiveThirtyEight)
│
├── models/
│   ├── poisson_model.py      # Dixon-Coles Poisson Goal Expectancy Model
│   └── ml_model.py           # XGBoost Multi-Output Goal Regressor dengan Count-Poisson Objective
│
├── value_bet.py              # Positive Expected Value (+EV) Finder & Fractional Kelly Stake Sizing
├── live_odds.py              # Ingestor live bookmaker odds (Pinnacle, Bet365) & consensus feed
├── sync_live_odds.py         # CLI & cron auto-sync jadwal & odds langsung ke SQLite ledger
│
├── tracker_db.py             # SQLite Manager (data/predictions_ledger.db) untuk audit trail pre-kickoff
├── tracking_service.py       # Settlement Service untuk rekonsiliasi skor aktual & perhitungan ROI / P&L
├── settle_predictions.py     # CLI audit tool (--auto, --report, manual settlement)
│
├── evaluator.py              # Out-of-Time Backtesting Harness (Log-Loss, Brier Score, MAE, RMSE)
├── run_backtest.py           # CLI runner evaluasi model & visualisasi chart
├── predictor.py              # Unified Match Inference Service (Poisson vs ML)
│
├── api.py                    # Production FastAPI REST Endpoints (port 8000)
├── app.py                    # Streamlit Interactive Web Dashboard & Radar (port 8501)
│
├── Dockerfile                # Multi-stage production container image
├── docker-compose.yml        # Orchestration (FastAPI, Streamlit, Scheduler Worker, Persistent Volume)
├── .dockerignore             # Exclusion rules untuk lean build context
├── start_docker.bat          # Windows 1-click Docker launcher
├── start_docker.sh           # Linux/macOS 1-click Docker launcher
├── requirements.txt          # Pinned dependencies
├── football_clean_data.csv   # Dataset historis bersih lintas 6 liga Eropa
├── evaluation_metrics.json   # Ringkasan metrik backtest terbaru
└── evaluation_charts.png     # Grafik diagnostik backtest terbaru
```

---

## 🧩 Komponen Utama

### Menyiapkan Python di Windows
- Jalankan `setup_env.bat` sekali untuk membuat `.venv` bila belum ada dan memasang paket dari `requirements.txt`.
- Buka folder proyek ini di VS Code. `.vscode/settings.json` memilih `.venv\Scripts\python.exe` sebagai interpreter default dan mengaktifkannya di terminal baru.
- Jalankan dashboard/API dengan `start_dashboard.bat` atau `start_api.bat`; kedua launcher memanggil Python dari `.venv` secara langsung agar tidak bergantung pada Python global atau aktivasi terminal.
- Jika VS Code masih memakai interpreter lain, pilih **Python: Select Interpreter** lalu pilih `.venv\Scripts\python.exe`.

### 1. Dixon-Coles Poisson Correction ([`models/poisson_model.py`](file:///c:/Users/Hype%20G12/.gemini/antigravity-ide/scratch/football-data-pipeline/models/poisson_model.py))
- Mengatasi kelemahan Poisson standar yang mengasumsikan skor tim independen.
- Parameter korelasi Dixon-Coles diestimasi dari data pelatihan pada setiap proses fit.
- Mengoreksi probabilitas untuk skor rendah; kualitas prediksi harus dinilai dengan backtest terbaru, bukan satu angka yang dianggap berlaku untuk semua musim.

### Simulasi Monte Carlo untuk Skenario
- Tab prediksi menyediakan simulasi Monte Carlo (default 20.000 iterasi, dapat dipilih 1.000–100.000) dengan peluang 1X2, skor yang paling sering tersimulasi, dan hasil per skenario.
- Masukkan bobot kemungkinan lineup/kondisi tim dan pengali xG kandang/tandang. Bobot dinormalisasi; pengali mengubah estimasi gol dasar model. Dampak pemain tidak dipelajari otomatis dan asumsi harus berasal dari pengguna.
- Ketidakpastian xG menggunakan variasi lognormal yang menjaga rata-rata xG dasar; CV default 10% adalah asumsi stress-test yang bisa diubah, bukan confidence interval empiris.
- Bila Dixon-Coles dipilih, simulasi mengoreksi peluang skor rendah memakai rho model. Interval 95% yang ditampilkan hanya mengukur galat sampling Monte Carlo, bukan rentang pasti skor atau ketidakpastian model.

### 2. Database Tracking & Settlement Ledger ([`tracker_db.py`](file:///c:/Users/Hype%20G12/.gemini/antigravity-ide/scratch/football-data-pipeline/tracker_db.py))
- Database SQLite thread-safe (`data/predictions_ledger.db`) mencatat setiap prediksi pra-laga secara *immutable* sebelum kickoff.
- Rekonsiliasi otomatis pasca-laga (`settle_predictions.py --auto`) menghitung:
  - Realized Profit & Loss (P&L Units)
  - Return on Investment (ROI %)
  - Closing Line Value (CLV %)
  - Dynamic Bankroll Curve (terintegrasi di Tab 4 Streamlit).

### 3. World Football Dynamic Elo Rating Engine ([`elo.py`](file:///c:/Users/Hype%20G12/.gemini/antigravity-ide/scratch/football-data-pipeline/elo.py))
- Sistem rating dinamis berbasis standar FIFA & FiveThirtyEight:
  - Base rating 1500, penyesuaian keunggulan kandang +65 Elo.
  - Dampak rating meningkat bertahap sesuai selisih gol.
- Rating semua pertandingan pada tanggal yang sama dihitung dari snapshot pra-pertandingan yang sama karena waktu kickoff historis tidak selalu tersedia.

### 4. Live Odds & Fixture Auto-Sync ([`live_odds.py`](file:///c:/Users/Hype%20G12/.gemini/antigravity-ide/scratch/football-data-pipeline/live_odds.py), [`sync_live_odds.py`](file:///c:/Users/Hype%20G12/.gemini/antigravity-ide/scratch/football-data-pipeline/sync_live_odds.py))
- Mengambil odds bookmaker terkini hanya dari *The Odds API*. Endpoint mengembalikan HTTP 503 / CLI berhenti dengan pesan jelas jika `ODDS_API_KEY` tidak tersedia atau feed gagal; tidak ada odds sintetis maupun cache stale.
- Tab Value Betting mengambil odds live untuk fixture terpilih dan menampilkan bookmaker serta waktu pengambilan. Odds manual pada `/value-bet` harus berasal dari harga bookmaker terkini; API tidak dapat memverifikasi harga yang dikirim pemanggil. Probabilitas model dan EV adalah estimasi, bukan jaminan skor atau profit.
- Otomatis memindai laga akhir pekan untuk menemukan **Positive Expected Value (+EV)**:
  $$\text{EV} = (P_{\text{model}} \times \text{Odds}_{\text{pasar}}) - 1$$
- Menghitung stake terukur menggunakan **Fractional Kelly Criterion** untuk perlindungan modal.
- Fitur **1-Click Commit All +EV Bets** di Tab 3 dashboard atau CLI otomatis via cron.

### 5. Contextual Football Features: Rest Days & Venue Fortress ([`features.py`](file:///c:/Users/Hype%20G12/.gemini/antigravity-ide/scratch/football-data-pipeline/features.py))
- **Squad Fatigue & Rest Days:** Menghitung jeda kalender istirahat antar laga ($T_i - T_{i-1}$) dan indikator jadwal padat (`is_congested`).
- **Venue-Specific Form:** Memisahkan statistik performa kandang (*Home Fortress*) dan tandang (*Away Travel Factor*).
- Fitur rolling dan rata-rata liga menggunakan data yang tersedia sebelum pertandingan; uji di masa depan harus tetap dijalankan ulang setelah perubahan data atau kode.

### Verifikasi Fixture dan Lineup Manual
- Tab prediksi dapat memverifikasi fixture, status pertandingan, waktu kick-off, dan skor akhir melalui football-data.org. Atur `FOOTBALL_DATA_API_TOKEN` (tersedia pada paket gratis) untuk mengaktifkannya; tanpa token, prediksi lokal tetap dapat digunakan tetapi verifikasi provider tidak tersedia.
- Provider ini tidak menyediakan susunan pemain. Tab prediksi menyediakan kolom lineup manual per tim; lineup hanya menjadi konteks sesi dashboard dan **tidak mengubah skor model**. Integrasi ke formula prediksi memerlukan data historis lineup dan backtest untuk membuktikan dampaknya.
- Verifikasi tersedia untuk EPL, La Liga, Bundesliga, Serie A, dan Ligue 1. Liga di luar cakupan ini, termasuk RFPL, tidak diklaim didukung. Cache fixture mencegah pemanggilan berulang yang tidak perlu.

### 6. Provenance & Ketersediaan Statistik
- Dataset historis saat ini diperbarui dari feed pertandingan Understat; kolom `ResultSource` dan `xGSource` menyimpan sumber tiap jenis data.
- Understat tidak menyediakan statistik tembakan melalui endpoint liga yang dipakai pipeline. Karena itu `HS`, `AS`, `HST`, dan `AST` dibiarkan kosong, bukan dianggap nol; model juga tidak memakai fitur tembakan yang tidak tersedia.
- Parser Football-Data.co.uk tetap tersedia sebagai feed alternatif. Nilai hasil/xG/statistik dipilih per jenis data saat fixture digabung, dan cache hanya boleh dipakai untuk musim yang sama.
- Endpoint `/fixtures/upcoming` menyajikan snapshot lokal Understat beserta waktu pembaruan dan tanggal akhir data model. `/fixtures/sample` adalah data contoh buatan dan tidak untuk keputusan taruhan.

### 7. Docker Containerization ([`Dockerfile`](file:///c:/Users/Hype%20G12/.gemini/antigravity-ide/scratch/football-data-pipeline/Dockerfile), [`docker-compose.yml`](file:///c:/Users/Hype%20G12/.gemini/antigravity-ide/scratch/football-data-pipeline/docker-compose.yml))
- Multi-container orchestration siap produksi:
  - **FastAPI Backend (Port 8000)** dengan healthcheck terpasang.
  - **Streamlit Web Dashboard (Port 8501)** dengan real-time radar visual.
  - **Scheduler Worker Daemon** (opsional) untuk auto-sync odds berkala.
  - **Named Volume `pipeline_data`** untuk memastikan database SQLite persisten dan tidak hilang saat container restart.

---

## 📊 Hasil Evaluasi Model

Hasil berikut dibuat dengan `python run_backtest.py --train-ratio 0.8` dari dataset lokal saat dokumentasi diperbarui. Pemisahan dilakukan berdasarkan tanggal penuh sehingga pertandingan pada hari yang sama tidak tersebar di train dan test.

| Model | 1X2 Accuracy | Exact Score Acc | Log-Loss | Brier Score | Goal MAE (H / A) | Goal RMSE (H / A) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Historical Mode)** | 42.74% | 11.69% | 1.0769 | 0.6520 | 1.04 / 0.89 | 1.26 / 1.11 |
| **Dixon-Coles Poisson** | 49.35% | 12.50% | 1.0301 | 0.6158 | 0.98 / 0.89 | 1.22 / 1.11 |
| **XGBoost Regressor + Context** | **50.48%** | **13.06%** | **1.0085** | **0.6021** | **0.96 / 0.84** | **1.18 / 1.06** |

Backtest ini memakai **5.058** pertandingan untuk training dan **1.240** untuk pengujian, pada periode **2026-01-18 sampai 2026-09-20**. Akurasi skor tepat XGBoost hanya 13.06%; skor dengan probabilitas tertinggi bukan jaminan skor aktual. Hasil dapat berubah saat dataset diperbarui. Dashboard membaca angka terbaru dari `evaluation_metrics.json`, yang dibuat bersama grafik oleh perintah backtest.

---

## 🐳 Panduan Menjalankan dengan Docker

### 1. Menjalankan API & Dashboard (Standar)
```bash
# Build dan jalankan di background:
docker compose up -d --build

# Buka di browser:
# - Streamlit Web Dashboard : http://localhost:8501
# - FastAPI Swagger Docs    : http://localhost:8000/docs
```

### 2. Menjalankan Lengkap Bersama Background Sync Daemon
```bash
# Menjalankan API + Dashboard + Background Scheduler:
docker compose --profile full up -d --build
```

### 3. Menghentikan Container
```bash
docker compose down
```
*(Catatan: Data di `pipeline_data` tetap tersimpan aman di host).*

## ☁️ Deploy Gratis Dashboard Streamlit

InfinityFree tidak menjalankan Python, Streamlit, Docker, atau FastAPI. Untuk deploy gratis dashboard, gunakan [Streamlit Community Cloud](https://share.streamlit.io/) dengan repository GitHub:

1. Pastikan repository berisi `app.py`, `requirements.txt`, `football_clean_data.csv`, `evaluation_metrics.json`, `evaluation_charts.png`, dan berkas data dashboard di folder `data/` (termasuk `players_database.csv` dan `upcoming_european_fixtures.json` jika fitur terkait ingin digunakan).
2. Buat aplikasi baru di Streamlit Community Cloud dan pilih repository, branch, serta file utama `app.py`.
3. Pilih Python 3.11 jika diminta (sama dengan runtime Docker proyek), lalu deploy. Dependensi akan dipasang dari `requirements.txt`.
4. Jika memakai token API, buka **App settings → Secrets** dan masukkan:

   ```toml
   ODDS_API_KEY = "token-the-odds-api"
   FOOTBALL_DATA_API_TOKEN = "token-football-data-org"
   ```

   Kedua token bersifat opsional; tanpa token, prediksi lokal tetap bisa berjalan, tetapi live odds dan verifikasi fixture dari provider terkait tidak tersedia. Jangan commit token atau `.streamlit/secrets.toml`.

Dashboard tetap menghitung prediksi Poisson/XGBoost, simulasi, dan analisis berdasarkan dataset yang disertakan. Namun, deploy ini **tidak** menjalankan FastAPI/API docs maupun scheduler otomatis. SQLite ledger berada di filesystem instance gratis yang tidak persisten; catatan P&L/prediksi dapat hilang saat aplikasi direstart atau dibangun ulang. Untuk API publik, scheduler, atau ledger yang persisten, gunakan layanan Python berbayar/persisten atau hosting VPS yang mengizinkan proses Python.

---

## 💻 Panduan Menjalankan Secara Lokal (Tanpa Docker)

### 1. Setup Virtual Environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

Untuk verifikasi fixture, atur `FOOTBALL_DATA_API_TOKEN` pada terminal yang menjalankan Streamlit atau FastAPI, atau isi di `.env` untuk Docker Compose. Di PowerShell, gunakan `$env:FOOTBALL_DATA_API_TOKEN="token-anda"` sebelum menjalankan proses. Dapatkan token melalui [football-data.org](https://www.football-data.org/client/register). Jangan simpan token asli di `.env.example` atau unggah ke repositori. Lineup tetap dimasukkan manual.

### 2. Menjalankan Dashboard Streamlit
```bash
streamlit run app.py
```

### 3. Menjalankan FastAPI REST Backend
```bash
uvicorn api:app --reload --port 8000
```

### 4. Menjalankan Prediksi Laga Tunggal
```bash
python run_prediction.py --home "Arsenal" --away "Chelsea" --compare
```

### 5. Memperbarui Data dan Prediksi Fixture Nyata
```bash
python update_all_european_data.py
```
Perintah ini mengambil data Understat terbaru dan hanya mengganti snapshot fixture serta CSV prediksi jika seluruh fixture berhasil diprediksi. API menolak snapshot fixture yang lebih tua dari 24 jam.

### 6. Memindai Live Odds & Commit ke SQLite Ledger
```bash
python sync_live_odds.py --commit-to-db --min-ev 5.0
```

### 7. Rekonsiliasi & Settle Hasil Pertandingan
```bash
# Settle semua pertandingan pending:
python settle_predictions.py --auto

# Cetak ringkasan performa finansial:
python settle_predictions.py --report
```

### 8. Menjalankan Ulang Backtest Out-of-Time
```bash
python run_backtest.py --train-ratio 0.8
```
Perintah tersebut memperbarui `evaluation_metrics.json` dan `evaluation_charts.png`; dashboard menampilkan metrik dari file hasil tersebut, bukan angka yang ditulis permanen di kode.

Prediksi adalah distribusi probabilitas, bukan jaminan skor benar 100%. Model default XGBoost dipilih karena mengungguli Poisson pada backtest out-of-time proyek ini; evaluasi ulang pada data baru karena keunggulan model dapat berubah. Estimasi skor mencakup ekor distribusi di atas 5 gol; model Poisson memakai baseline gol per liga, dan fitur prediksi hanya memakai pertandingan sebelum tanggal laga. Sertakan tanggal laga agar form dan waktu istirahat dihitung sesuai fixture:

```bash
curl "http://127.0.0.1:8000/predict?home_team=Arsenal&away_team=Chelsea&method=poisson&match_date=2026-10-10"
```

Ekspor `run_matchday_forecast.py` menyimpan tanggal/ID fixture, waktu prediksi, batas tanggal data, dan SHA-256 dataset untuk menelusuri asal prediksi. Script tidak lagi membuat prediksi dari daftar fixture contoh jika sumber fixture nyata tidak tersedia.

---

## 🔒 Integritas Data & Keamanan

- **Point-in-time features:** Statistik rolling, rata-rata gol liga, dan rating Elo dihitung hanya dari laga sebelum tanggal pertandingan; pertandingan pada tanggal yang sama tidak saling memperbarui rating Elo.
- **Deduplikasi Cerdas:** Proteksi duplikasi entri prediksi berbasis kombinasi `(HomeTeam, AwayTeam, Date)` dengan status `PENDING`.
- **Non-root Container User:** Kontainer Docker dijalankan dengan UID `1000` (`appuser`) untuk kepatuhan standar keamanan cloud native.
