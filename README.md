# pbp-anomaly

Training-free anomaly detection in environmental sensor networks via pseudo-Boolean polynomial (PBP) decomposition.

Companion code for:

> T. Chikake and B. Goldengorin, "Training-free environmental anomaly detection with sensor-pair diagnostics," *Nature*, 2026. (Under review)

## Quick start

```bash
git clone https://github.com/Tenfleques/pbp-anomaly.git
cd pbp-anomaly
pip install -e ".[all]"

# Download large datasets (Beijing, NOAA weather stations)
python experiments/download_data.py

# Verify pre-computed results (should report 200+ PASS, 0 FAIL)
python experiments/verify_consistency.py

# Generate all figures
python experiments/generate_figures.py
```

## What this repo contains

- `pbp_anomaly/` -- Python package implementing the PBP anomaly detector
- `data/` -- Small datasets shipped with the repo (UCI, OpenMeteo); large datasets downloaded via script
- `results/precomputed/` -- Pre-computed experiment results (JSON/CSV) for instant verification
- `experiments/` -- Scripts to reproduce all results and figures from the paper
- `tests/` -- Unit tests (27 tests)

## Datasets

The paper evaluates PBP across 9 datasets spanning two sensor domains:

| Dataset | Domain | Sensors | Readings | Source | Shipped |
|---------|--------|---------|----------|--------|---------|
| UCI Air Quality (Italy) | Air quality | 8 | 9,357 | [UCI MLR](https://archive.ics.uci.edu/dataset/360/air+quality) | Yes |
| Beijing Dongsi | Air quality | 10 | 35,064 | [Zhang et al. 2017](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) | Download |
| EPA AQS (Los Angeles) | Air quality | 4 | 26,280 | [EPA AQS](https://aqs.epa.gov/aqsweb/airdata/download_files.html) | Optional |
| Sao Paulo (CAMS) | Air quality | 6 | 17,520 | [Open-Meteo](https://open-meteo.com/en/docs/air-quality-api) | Yes |
| Cape Town (CAMS) | Air quality | 6 | 17,520 | [Open-Meteo](https://open-meteo.com/en/docs/air-quality-api) | Yes |
| Chicago O'Hare | Weather | 6 | 17,520 | [NOAA ISD](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) | Download |
| Miami | Weather | 6 | 17,520 | [NOAA ISD](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) | Download |
| San Francisco | Weather | 6 | 17,520 | [NOAA ISD](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) | Download |
| Fairbanks | Weather | 6 | 17,520 | [NOAA ISD](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) | Download |

"Shipped" means the CSV is included in this repo. "Download" means `download_data.py` fetches it automatically.

## Reproducing results

### Verify pre-computed results

```bash
python experiments/verify_consistency.py
```

This checks every number cited in the manuscript against the pre-computed result files in `results/precomputed/`. No data download required.

### Full replication from raw data

```bash
# Download datasets
python experiments/download_data.py

# Run all experiments (may take several hours)
python experiments/replicate_all.py --output-dir results/local

# Verify fresh results
python experiments/verify_consistency.py --results-dir results/local

# Generate figures from fresh results
python experiments/generate_figures.py --results-dir results/local
```

For a quick smoke test on a single dataset:

```bash
python experiments/replicate_all.py --datasets UCI --quick --output-dir results/local
```

### Figures

```bash
python experiments/generate_figures.py
```

Generates Fig 1 (temporal robustness), Fig 2 (hub variable heatmap), and Fig 3 (sensor-pair z-score case study) in `figures/`.

## Library usage

```python
from pbp_anomaly import AnomalyDetector
import pandas as pd

df = pd.read_csv('your_sensor_data.csv')
detector = AnomalyDetector(window_size=6, mode='both')
result = detector.fit_score(df, sensor_cols=['CO', 'NO2', 'O3'])

print(f"AUC-ROC: {result['standard']['auc_roc']:.3f}")
print(f"Top sensor pairs: {result['pair_ranking'][:3]}")
```

Or via CLI:

```bash
pbp-anomaly detect data.csv --sensors CO,NO2,O3 --mode both
```

## Running tests

```bash
pytest tests/ -v
```

## Dependencies

Core: numpy, pandas, scipy, scikit-learn, [tmc-pbp](https://github.com/Tenfleques/tmc-pbp)

Optional: matplotlib (figures), requests (NOAA download), pytest (tests)

All installed automatically via `pip install -e ".[all]"`.

## Citation

```bibtex
@article{Chikake2026anomaly,
  author  = {Chikake, Tendai and Goldengorin, Boris},
  title   = {Training-free environmental anomaly detection with sensor-pair diagnostics},
  journal = {Nature},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT
