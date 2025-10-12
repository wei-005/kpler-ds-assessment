
# Kpler DS case study 

## What's inside
- `notebooks/01_...` to `06_...`: end‑to‑end jupyter noetbook.
- `utils/`: small library the notebooks import:
  - `config.py`: path config helpers
  - `etl_clean.py`: cleaning + Task‑A samples builder (keeps `is_load/is_discharge`, `prev_dist_km`, `last_leg_knots_est`, `product_family_dom`)
  - `splits.py`: temporal split + crisis flag
  - `candidates.py`: transitions, global MF, history/geo candidates
  - `features.py`: port attributes, port degree, sample features, merge
  - `metrics.py`: hits@K/MRR, candidate recall

## How to use
1. Place this folder at your project root (or unzip it there).
2. Set data paths in **notebooks/01** (or via env vars `KPLER_DATA_DIR`, `KPLER_INTERIM_DIR`, `KPLER_PROCESSED_DIR`).
3. Run notebooks in order 01 → 06.

## Notes
- focused on **Task A (very next destination)** per assignment.
- we can later modularize logic further into `utils` if needed.
