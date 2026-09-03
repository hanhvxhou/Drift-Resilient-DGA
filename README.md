# DRC-CL: Drift-Resilient Continual Learning for DGA Botnet Detection

## Project structure

```
drc_cl_project/
│
├── configs/
│   └── config.yaml              ← all hyperparameters & paths (edit this)
│
├── data/
│   ├── raw/
│   │   ├── dgarchive/           ← PUT *_dga.csv files HERE
│   │   └── benign/              ← PUT tranco_YYYY.csv / alexa_YYYY.csv HERE
│   ├── interim/                 ← auto-generated (merged Parquet)
│   └── processed/
│       ├── windows/             ← D01_dga.csv … D24_dga.csv  (DGA only)
│       └── benchmark/           ← D01.csv … D24.csv  (DGA + benign, final)
│                                   drift_labels.json
│                                   integrity_report.txt
│
├── src/
│   ├── data/
│   │   ├── step1_merge_dgarchive.py   ← read CSVs → Parquet
│   │   ├── step2_build_dga_windows.py ← 24 quarterly windows (DGA)
│   │   ├── step3_merge_benign.py      ← add benign, cross-contamination check
│   │   ├── step4_annotate_drift.py    ← MMD² drift labels
│   │   ├── step5_integrity_report.py  ← final QA
│   │   └── run_pipeline.py            ← master runner
│   ├── models/                        ← CharCNN backbone (separate module)
│   ├── detect/                        ← ADD drift detector (separate module)
│   └── utils/
│       └── common.py                  ← config, logging, quarter helpers
│
├── notebooks/
│   └── (EDA notebooks go here)
│
├── tests/
│   └── test_data_pipeline.py          ← pytest unit + integration tests
│
├── results/
│   ├── checkpoints/
│   ├── logs/
│   └── figures/
│
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Set up environment

```bash
cd drc_cl_project
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Place raw data

```
data/raw/dgarchive/
    blackhole_dga.csv
    conficker_dga.csv
    necurs_dga.csv
    gameover_p2p_dga.csv
    ...   (all *_dga.csv from DGArchive)

data/raw/benign/
    tranco_2018.csv     ← single column: domain
    tranco_2019.csv
    ...
    tranco_2023.csv
    alexa_2018.csv      ← optional fallback, same format
    ...
```

CSV format for DGArchive files (as provided):
```
domain,domain_id,valid_from,valid_to,seed_id
"abc123.com","7263","2018-02-19 00:00:00","2018-02-19 23:59:59","gameover_dga_seed"
```

CSV format for benign files (one per year):
```
domain
google.com
github.com
...
```

### 3. Run the full pipeline

```bash
# From the project root (drc_cl_project/)
# With stub drift labels (no backbone needed):
python -m src.data.run_pipeline --stub

# With backbone for real drift labels:
python -m src.data.run_pipeline --backbone results/checkpoints/backbone_d01.pt

# Resume from step 3 (skip already-done steps):
python -m src.data.run_pipeline --stub --start-from 3
```

### 4. Run individual steps

```bash
python -m src.data.step1_merge_dgarchive   # → data/interim/dgarchive_merged.parquet
python -m src.data.step2_build_dga_windows # → data/processed/windows/D01_dga.csv…
python -m src.data.step3_merge_benign      # → data/processed/benchmark/D01.csv…
python -m src.data.step4_annotate_drift --stub  # → drift_labels.json
python -m src.data.step5_integrity_report  # → integrity_report.txt
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Output file format

Each benchmark window file (`benchmark/D01.csv` … `D24.csv`) contains:

| Column         | Type   | Description                              |
|----------------|--------|------------------------------------------|
| `domain`       | str    | Domain name string                       |
| `label`        | int    | 1 = DGA (malicious), 0 = benign          |
| `family`       | str    | DGA family name, or "benign"             |
| `quarter_id`   | str    | Window ID: D01 … D24                     |
| `quarter_label`| str    | Human-readable: 2018_Q1 … 2023_Q4        |

`drift_labels.json` structure:
```json
{
  "boundaries": [
    {
      "boundary_id": "D01_to_D02",
      "window_from": "D01",
      "window_to":   "D02",
      "mmd2":        0.00341,
      "cosine_sim_to_history": 0.12,
      "drift_type":  "gradual"
    },
    ...
  ],
  "thresholds": {"delta1": 0.05, "delta2": 0.01, "tau": 0.85, "k": 3}
}
```

Drift types: `none` | `gradual` | `sudden` | `recurring` | `UNLABELED` (stub)

---

## Configuration

All parameters are in `configs/config.yaml`. Key settings:

```yaml
dga:
  min_family_samples: 2000   # families with fewer total samples are excluded
  max_per_window:    50000   # DGA domains sampled per quarter (stratified)

benign:
  max_per_window:  50000     # benign domains sampled per quarter

random_seed: 42              # controls all sampling for reproducibility
```

---

## Reproducibility

All sampling is controlled by `random_seed` in `config.yaml`.
To exactly reproduce the paper benchmark, use `random_seed: 42`.
Run logs are saved to `results/logs/`.

---

## PyCharm setup

1. Open `drc_cl_project/` as the project root in PyCharm.
2. Set Project Interpreter to the `.venv` virtual environment.
3. Mark `drc_cl_project/` as "Sources Root" (right-click → Mark Directory As → Sources Root).
   This ensures `from src.data.xxx import ...` resolves correctly.
4. Run configurations: use "Module" mode (`python -m src.data.run_pipeline`).
