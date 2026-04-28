#!/usr/bin/env python3
"""
Unified replication script for all Nature article experiments.

Reproduces all weather + air quality anomaly detection results using
the pbp_anomaly module. Outputs a single JSON with the full
cross-domain comparison matrix.

Usage:
    python replicate_all.py                    # Run all datasets
    python replicate_all.py --datasets UCI MIA # Run specific datasets
    python replicate_all.py --quick            # w=6 only, no sweep
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Ensure pbp_anomaly is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pbp_anomaly.detector import (
    AnomalyDetector, compute_standard_features, compute_pair_features,
    create_labels, feature_aucs, composite_score, bootstrap_ci, get_backend,
)
from pbp_anomaly.parsers import (
    load_noaa_isd, load_uci_air_quality, load_beijing_multisite,
    load_open_meteo_csv,
)

# ─── Dataset registry ──────────────────────────────────────────────────

NATURE_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'publications',
    'articles-2026', 'PBP-environmental-anomaly-nature', 'experiments', 'data'
))

DATASETS = {
    # Air quality
    'UCI': {
        'loader': lambda: load_uci_air_quality(
            os.path.join(NATURE_DATA_DIR, 'AirQualityUCI.csv')),
        'domain': 'air_quality',
    },
    'Beijing': {
        'loader': lambda: load_beijing_multisite(
            os.path.join(NATURE_DATA_DIR, 'beijing_multisite')),
        'domain': 'air_quality',
    },
    'SaoPaulo': {
        'loader': lambda: load_open_meteo_csv(
            os.path.join(NATURE_DATA_DIR, 'open_meteo_são_paulo', 'air_quality.csv'),
            'São Paulo'),
        'domain': 'air_quality',
    },
    'CapeTown': {
        'loader': lambda: load_open_meteo_csv(
            os.path.join(NATURE_DATA_DIR, 'open_meteo_cape_town', 'air_quality.csv'),
            'Cape Town'),
        'domain': 'air_quality',
    },
    # Weather stations
    'ORD': {
        'loader': lambda: load_noaa_isd(
            os.path.join(NATURE_DATA_DIR, 'noaa_isd_ord'), 'ord',
            station_name="Chicago O'Hare (continental)"),
        'domain': 'weather',
    },
    'MIA': {
        'loader': lambda: load_noaa_isd(
            os.path.join(NATURE_DATA_DIR, 'noaa_isd_mia'), 'mia',
            station_name='Miami (tropical)'),
        'domain': 'weather',
    },
    'SFO': {
        'loader': lambda: load_noaa_isd(
            os.path.join(NATURE_DATA_DIR, 'noaa_isd_sfo'), 'sfo',
            station_name='San Francisco (maritime)'),
        'domain': 'weather',
    },
    'FAI': {
        'loader': lambda: load_noaa_isd(
            os.path.join(NATURE_DATA_DIR, 'noaa_isd_fai'), 'fai',
            station_name='Fairbanks (subarctic)'),
        'domain': 'weather',
    },
}

# EPA requires symlink to NFS — optional
try:
    epa_dir = os.path.join(NATURE_DATA_DIR, 'epa_aqs_2023')
    if os.path.isdir(epa_dir) or os.path.islink(epa_dir):
        # EPA uses its own loader from the original pipeline (ZIP-based)
        # For replication, we import from the original experiment if available
        _exp_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'publications',
            'articles-2026', 'PBP-environmental-anomaly-nature', 'experiments'))
        sys.path.insert(0, _exp_dir)
        from multi_dataset_anomaly import load_epa_aqs as _load_epa

        # Patch os.makedirs for symlink compatibility
        _orig_makedirs = os.makedirs
        def _safe_makedirs(name, mode=0o777, exist_ok=False):
            if os.path.islink(name) or os.path.isdir(name):
                if exist_ok:
                    return
            _orig_makedirs(name, mode, exist_ok=exist_ok)
        os.makedirs = _safe_makedirs

        DATASETS['EPA_LA'] = {
            'loader': lambda: _load_epa(NATURE_DATA_DIR, year=2023),
            'domain': 'air_quality',
        }
        os.makedirs = _orig_makedirs
except Exception:
    pass


# ─── Baselines ──────────────────────────────────────────────────────────

def run_baselines(values_sc, labels, window_size, n_sensors):
    """Run IF, OC-SVM, LOF baselines."""
    n_win = len(labels)
    flat = np.zeros((n_win, window_size * n_sensors))
    for i in range(n_win):
        flat[i] = values_sc[i:i + window_size].flatten()

    normal = flat[labels == 0]
    X_train = normal[:int(0.7 * len(normal))]
    contam = min(labels.mean(), 0.5)

    results = {}
    for name, model in [
        ('IF', IsolationForest(n_estimators=100, contamination=contam, random_state=42)),
        ('OC-SVM', OneClassSVM(kernel='rbf', nu=min(contam, 0.5))),
        ('LOF', LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=contam)),
    ]:
        model.fit(X_train)
        scores = -model.score_samples(flat)
        auc = roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else 0.5
        results[name] = {
            'auc_roc': round(float(auc), 4),
            'ci_95': bootstrap_ci(scores, labels),
        }
    return results


# ─── Per-dataset pipeline ──────────────────────────────────────────────

def run_dataset(ds, window_sizes=(6,), include_baselines=True):
    """Run full pipeline on one dataset."""
    name = ds['name']
    data = ds['data']
    sensor_cols = ds['sensor_cols']
    ref_cols = ds['ref_cols']
    n_sensors = len(sensor_cols)

    print(f"\n{'='*60}", flush=True)
    print(f"DATASET: {name} ({n_sensors} sensors, {len(data)} readings)", flush=True)
    print(f"{'='*60}", flush=True)

    values = data[sensor_cols].values

    result = {
        'dataset': name,
        'n_readings': len(data),
        'n_sensors': n_sensors,
        'backend': get_backend(),
    }

    for ws in window_sizes:
        print(f"\n  --- w={ws}, one-tailed ---", flush=True)
        labels = create_labels(data[ref_cols].values, ws)
        n_win = len(labels)
        af = labels.mean()
        print(f"  Windows: {n_win}, Anomalous: {labels.sum()} ({100*af:.1f}%)", flush=True)

        if len(np.unique(labels)) < 2:
            result[f'w{ws}'] = {'error': 'no anomalies'}
            continue

        scaler = StandardScaler()
        values_sc = scaler.fit_transform(values)

        # PBP standard
        t0 = time.perf_counter()
        feats_std = compute_standard_features(values, values_sc, ws)
        t_std = time.perf_counter() - t0

        aucs_std = feature_aucs(feats_std, labels)
        top5_std = list(np.argsort(aucs_std)[::-1][:5])
        comp_std = composite_score(feats_std, top5_std)
        auc_std = roc_auc_score(labels, comp_std)

        # PBP pair
        t0 = time.perf_counter()
        feats_pair, pairs = compute_pair_features(values, values_sc, n_sensors, ws)
        t_pair = time.perf_counter() - t0

        aucs_pair = feature_aucs(feats_pair, labels)
        top5_pair = list(np.argsort(aucs_pair)[::-1][:5])
        comp_pair = composite_score(feats_pair, top5_pair)
        auc_pair = roc_auc_score(labels, comp_pair)

        ws_result = {
            'n_windows': n_win,
            'anomaly_frac': round(float(af), 4),
            'pbp_std': {
                'auc_roc': round(float(auc_std), 4),
                'ci_95': bootstrap_ci(comp_std, labels),
                'time_s': round(t_std, 1),
            },
            'pbp_pair': {
                'auc_roc': round(float(auc_pair), 4),
                'ci_95': bootstrap_ci(comp_pair, labels),
                'time_s': round(t_pair, 1),
            },
        }

        # Pair ranking
        pair_ranking = []
        for p_idx, (s1, s2) in enumerate(pairs):
            pa = aucs_pair[p_idx * 12:(p_idx + 1) * 12]
            pair_ranking.append({
                'pair': f"{sensor_cols[s1]}–{sensor_cols[s2]}",
                'auc': round(float(pa.max()), 4),
            })
        pair_ranking.sort(key=lambda x: x['auc'], reverse=True)
        ws_result['pair_ranking'] = pair_ranking[:10]

        winner = 'std' if auc_std > auc_pair else 'pair'
        gap = abs(auc_std - auc_pair)
        print(f"  PBP std:  {auc_std:.4f}  ({t_std:.0f}s)", flush=True)
        print(f"  PBP pair: {auc_pair:.4f}  ({t_pair:.0f}s)", flush=True)
        print(f"  Winner: {winner} (+{gap:.3f})", flush=True)

        # Baselines
        if include_baselines:
            bl = run_baselines(values_sc, labels, ws, n_sensors)
            ws_result['baselines'] = bl
            for bn, br in bl.items():
                print(f"  {bn:<8s}  {br['auc_roc']:.4f}", flush=True)

        result[f'w{ws}'] = ws_result

        # Temporal split (only at primary window size)
        if ws == window_sizes[0]:
            print(f"\n  --- Temporal split (w={ws}) ---", flush=True)
            split = len(values) // 2
            tv, ev = values[:split], values[split:]
            sc2 = StandardScaler()
            sc2.fit(tv)
            tv_sc, ev_sc = sc2.transform(tv), sc2.transform(ev)
            tr_labels = create_labels(data[ref_cols].values[:split], ws)
            te_labels = create_labels(data[ref_cols].values[split:], ws)

            if len(np.unique(te_labels)) < 2:
                result['temporal_split'] = {'error': 'no test anomalies'}
            else:
                print(f"  Train: {len(tr_labels)} ({100*tr_labels.mean():.1f}%)", flush=True)
                print(f"  Test:  {len(te_labels)} ({100*te_labels.mean():.1f}%)", flush=True)

                f_std_tr = compute_standard_features(tv, tv_sc, ws)
                f_pair_tr, _ = compute_pair_features(tv, tv_sc, n_sensors, ws)
                f_std_te = compute_standard_features(ev, ev_sc, ws)
                f_pair_te, _ = compute_pair_features(ev, ev_sc, n_sensors, ws)

                a_s = feature_aucs(f_std_tr, tr_labels)
                a_p = feature_aucs(f_pair_tr, tr_labels)
                t5s = list(np.argsort(a_s)[::-1][:5])
                t5p = list(np.argsort(a_p)[::-1][:5])

                cs_te = composite_score(f_std_te, t5s)
                cp_te = composite_score(f_pair_te, t5p)
                auc_s_ts = roc_auc_score(te_labels, cs_te)
                auc_p_ts = roc_auc_score(te_labels, cp_te)

                print(f"  TS std:  {auc_s_ts:.4f}", flush=True)
                print(f"  TS pair: {auc_p_ts:.4f}", flush=True)

                result['temporal_split'] = {
                    'pbp_std': round(float(auc_s_ts), 4),
                    'pbp_pair': round(float(auc_p_ts), 4),
                }

    return result


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Replicate all Nature anomaly experiments')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to run (default: all)')
    parser.add_argument('--quick', action='store_true',
                        help='w=6 only, skip window sweep')
    parser.add_argument('--output', '-o', default='replication_results.json',
                        help='Output JSON path')
    args = parser.parse_args()

    window_sizes = (6,) if args.quick else (6, 12, 24)
    targets = args.datasets or list(DATASETS.keys())

    print(f"PBP backend: {get_backend()}", flush=True)
    print(f"Datasets: {targets}", flush=True)
    print(f"Window sizes: {window_sizes}", flush=True)
    print(f"{'='*60}", flush=True)

    all_results = {}
    for key in targets:
        if key not in DATASETS:
            print(f"Unknown dataset: {key}. Available: {list(DATASETS.keys())}")
            continue
        try:
            ds = DATASETS[key]['loader']()
            ds_result = run_dataset(ds, window_sizes=window_sizes)
            ds_result['domain'] = DATASETS[key]['domain']
            all_results[key] = ds_result
        except Exception as e:
            print(f"\n  ERROR on {key}: {e}", flush=True)
            all_results[key] = {'error': str(e)}

    # Save
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("CROSS-DOMAIN COMPARISON MATRIX", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Dataset':<30s} {'Domain':<12s} {'Sens':>4s} {'PBP-std':>8s} "
          f"{'PBP-pair':>9s} {'Win':>5s} {'TS-std':>7s} {'TS-pair':>8s}", flush=True)
    print('-' * 90, flush=True)
    for key in targets:
        r = all_results.get(key, {})
        if 'error' in r:
            print(f"  {key}: ERROR — {r['error'][:40]}", flush=True)
            continue
        w6 = r.get('w6', {})
        if 'error' in w6:
            continue
        s = w6['pbp_std']['auc_roc']
        p = w6['pbp_pair']['auc_roc']
        dom = r.get('domain', '?')
        ns = r['n_sensors']
        win = 'std' if s > p else 'pair'
        ts = r.get('temporal_split', {})
        ts_s = ts.get('pbp_std', '---')
        ts_p = ts.get('pbp_pair', '---')
        name = r['dataset']
        print(f"{name:<30s} {dom:<12s} {ns:>4d} {s:>8.4f} {p:>9.4f} "
              f"{win:>5s} {ts_s:>7} {ts_p:>8}", flush=True)


if __name__ == '__main__':
    main()
