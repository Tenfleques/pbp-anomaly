#!/usr/bin/env python3
"""
Unified replication script for all Nature article experiments.

Reproduces all weather + air quality anomaly detection results using
the pbp_anomaly module. Outputs multiple JSON files matching the schema
expected by verify_consistency.py.

Usage:
    python replicate_all.py                        # All datasets
    python replicate_all.py --datasets UCI MIA     # Specific datasets
    python replicate_all.py --quick                # w=6 only, no sweep
    python replicate_all.py --include-epa          # Include EPA dataset
    python replicate_all.py --output-dir results/local
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ─── Path setup ───────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PACKAGE_DIR)

from pbp_anomaly.detector import (
    compute_standard_features, compute_pair_features,
    create_labels, feature_aucs, composite_score, bootstrap_ci, get_backend,
    N_FEATURES,
)
from pbp_anomaly.parsers import (
    load_noaa_isd, load_uci_air_quality, load_beijing_multisite,
    load_open_meteo_csv, load_epa_aqs,
)

DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))


# ─── Dataset registry ────────────────────────────────────────────────

def _build_dataset_registry(include_epa=False):
    """Build dataset registry with loaders. EPA is optional."""
    registry = {
        # Air quality datasets
        'UCI': {
            'loader': lambda: load_uci_air_quality(
                os.path.join(DATA_DIR, 'AirQualityUCI.csv')),
            'domain': 'air_quality',
        },
        'Beijing': {
            'loader': lambda: load_beijing_multisite(
                os.path.join(DATA_DIR, 'beijing_multisite')),
            'domain': 'air_quality',
        },
        'SaoPaulo': {
            'loader': lambda: load_open_meteo_csv(
                os.path.join(DATA_DIR, 'open_meteo_sao_paulo.csv'),
                'São Paulo'),
            'domain': 'air_quality',
        },
        'CapeTown': {
            'loader': lambda: load_open_meteo_csv(
                os.path.join(DATA_DIR, 'open_meteo_cape_town.csv'),
                'Cape Town'),
            'domain': 'air_quality',
        },
        # Weather stations
        'ORD': {
            'loader': lambda: load_noaa_isd(
                os.path.join(DATA_DIR, 'noaa_isd_ord'), 'ord',
                station_name="Chicago O'Hare (continental)"),
            'domain': 'weather',
        },
        'MIA': {
            'loader': lambda: load_noaa_isd(
                os.path.join(DATA_DIR, 'noaa_isd_mia'), 'mia',
                station_name='Miami (tropical)'),
            'domain': 'weather',
        },
        'SFO': {
            'loader': lambda: load_noaa_isd(
                os.path.join(DATA_DIR, 'noaa_isd_sfo'), 'sfo',
                station_name='San Francisco (maritime)'),
            'domain': 'weather',
        },
        'FAI': {
            'loader': lambda: load_noaa_isd(
                os.path.join(DATA_DIR, 'noaa_isd_fai'), 'fai',
                station_name='Fairbanks (subarctic)'),
            'domain': 'weather',
        },
    }

    if include_epa:
        registry['EPA_LA'] = {
            'loader': lambda: load_epa_aqs(DATA_DIR, year=2023),
            'domain': 'air_quality',
        }

    return registry


# ─── Baseline methods ────────────────────────────────────────────────

def _flatten_windows(values_sc, window_size, n_windows):
    """Flatten sliding windows into (n_windows, window_size * n_sensors) array."""
    n_sensors = values_sc.shape[1]
    flat = np.zeros((n_windows, window_size * n_sensors))
    for i in range(n_windows):
        flat[i] = values_sc[i:i + window_size].flatten()
    return flat


def run_baselines(values_sc, labels, window_size, n_sensors):
    """Run Isolation Forest, One-Class SVM, and LOF baselines.

    Train on 70% of normal-labeled windows.
    Returns dict keyed by method name with auc_roc and ci_95.
    """
    n_win = len(labels)
    flat = _flatten_windows(values_sc, window_size, n_win)

    normal_mask = labels == 0
    normal_data = flat[normal_mask]
    n_train = int(0.7 * len(normal_data))
    X_train = normal_data[:n_train]

    contamination = min(float(labels.mean()), 0.5)

    models = [
        ('Isolation Forest', IsolationForest(
            n_estimators=100, contamination=contamination, random_state=42)),
        ('One-Class SVM', OneClassSVM(
            kernel='rbf', nu=min(contamination, 0.5))),
        ('LOF', LocalOutlierFactor(
            n_neighbors=20, novelty=True, contamination=contamination)),
    ]

    results = {}
    for name, model in models:
        model.fit(X_train)
        scores = -model.score_samples(flat)
        if len(np.unique(labels)) == 2:
            auc_val = roc_auc_score(labels, scores)
            ci = bootstrap_ci(scores, labels)
        else:
            auc_val = 0.5
            ci = [0.5, 0.5]
        results[name] = {
            'method': name,
            'auc_roc': round(float(auc_val), 4),
            'ci_95': ci,
        }

    return results


# ─── PBP pipeline for a single window size + label config ─────────

def _run_pbp_pipeline(values, values_sc, labels, n_sensors, ws, sensor_cols):
    """Run PBP standard + pair feature extraction and scoring.

    Returns a dict with pbp_std, pbp_pair results and pair_ranking.
    """
    result = {}

    # PBP standard features
    t0 = time.perf_counter()
    feats_std = compute_standard_features(values, values_sc, ws)
    t_std = time.perf_counter() - t0

    aucs_std = feature_aucs(feats_std, labels)
    top5_std = list(np.argsort(aucs_std)[::-1][:5])
    comp_std = composite_score(feats_std, top5_std)
    auc_std = roc_auc_score(labels, comp_std) if len(np.unique(labels)) == 2 else 0.5

    result['pbp_std'] = {
        'method': 'PBP standard',
        'auc_roc': round(float(auc_std), 4),
        'ci_95': bootstrap_ci(comp_std, labels),
    }
    result['time_std_s'] = round(t_std, 1)
    result['_comp_std'] = comp_std  # internal, stripped before saving
    result['_feats_std'] = feats_std
    result['_aucs_std'] = aucs_std
    result['_top5_std'] = top5_std

    # PBP pair features
    t0 = time.perf_counter()
    feats_pair, pairs = compute_pair_features(values, values_sc, n_sensors, ws)
    t_pair = time.perf_counter() - t0

    aucs_pair = feature_aucs(feats_pair, labels)
    top5_pair = list(np.argsort(aucs_pair)[::-1][:5])
    comp_pair = composite_score(feats_pair, top5_pair)
    auc_pair = roc_auc_score(labels, comp_pair) if len(np.unique(labels)) == 2 else 0.5

    result['pbp_pair'] = {
        'method': 'PBP pair',
        'auc_roc': round(float(auc_pair), 4),
        'ci_95': bootstrap_ci(comp_pair, labels),
    }
    result['time_pair_s'] = round(t_pair, 1)
    result['_comp_pair'] = comp_pair
    result['_feats_pair'] = feats_pair
    result['_aucs_pair'] = aucs_pair
    result['_top5_pair'] = top5_pair
    result['_pairs'] = pairs

    # Pair ranking: which sensor pairs have highest individual AUC
    pair_ranking = []
    for p_idx, (s1, s2) in enumerate(pairs):
        pa = aucs_pair[p_idx * 12:(p_idx + 1) * 12]
        pair_ranking.append({
            'pair': [sensor_cols[s1], sensor_cols[s2]],
            'pair_str': f"{sensor_cols[s1]}–{sensor_cols[s2]}",
            'auc': round(float(pa.max()), 4),
        })
    pair_ranking.sort(key=lambda x: x['auc'], reverse=True)
    result['pair_ranking'] = pair_ranking

    return result


def _run_temporal_split(values, ref_values, n_sensors, ws, sensor_cols):
    """Run temporal split: train on first half, test on second half.

    Returns dict with pbp_std and pbp_pair AUC on test set,
    plus baselines on the temporal split.
    """
    split = len(values) // 2
    train_vals = values[:split]
    test_vals = values[split:]

    scaler = StandardScaler()
    scaler.fit(train_vals)
    train_sc = scaler.transform(train_vals)
    test_sc = scaler.transform(test_vals)

    train_labels = create_labels(ref_values[:split], ws, percentile=95)
    test_labels = create_labels(ref_values[split:], ws, percentile=95)

    if len(np.unique(test_labels)) < 2:
        return {'error': 'no test anomalies'}

    result = {
        'train_anom_frac': round(float(train_labels.mean()), 4),
        'test_anom_frac': round(float(test_labels.mean()), 4),
    }

    # Train: compute features, select top-5
    f_std_tr = compute_standard_features(train_vals, train_sc, ws)
    f_pair_tr, _ = compute_pair_features(train_vals, train_sc, n_sensors, ws)
    a_std_tr = feature_aucs(f_std_tr, train_labels)
    a_pair_tr = feature_aucs(f_pair_tr, train_labels)
    top5_std = list(np.argsort(a_std_tr)[::-1][:5])
    top5_pair = list(np.argsort(a_pair_tr)[::-1][:5])

    # Test: compute features, apply train-selected top-5
    f_std_te = compute_standard_features(test_vals, test_sc, ws)
    f_pair_te, _ = compute_pair_features(test_vals, test_sc, n_sensors, ws)
    comp_std_te = composite_score(f_std_te, top5_std)
    comp_pair_te = composite_score(f_pair_te, top5_pair)

    auc_std_ts = roc_auc_score(test_labels, comp_std_te)
    auc_pair_ts = roc_auc_score(test_labels, comp_pair_te)

    result['pbp_std'] = {
        'method': 'PBP standard',
        'auc_roc': round(float(auc_std_ts), 4),
    }
    result['pbp_pair'] = {
        'method': 'PBP pair',
        'auc_roc': round(float(auc_pair_ts), 4),
    }

    # Baselines on temporal split
    n_win_te = len(test_labels)
    flat_te = _flatten_windows(test_sc, ws, n_win_te)
    flat_tr = _flatten_windows(train_sc, ws, len(train_labels))
    normal_tr = flat_tr[train_labels == 0]
    n_train = int(0.7 * len(normal_tr))
    X_train = normal_tr[:n_train]
    contamination = min(float(train_labels.mean()), 0.5)

    baselines = {}
    for name, model in [
        ('IF', IsolationForest(n_estimators=100, contamination=contamination, random_state=42)),
        ('OC-SVM', OneClassSVM(kernel='rbf', nu=min(contamination, 0.5))),
        ('LOF', LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=contamination)),
    ]:
        model.fit(X_train)
        scores = -model.score_samples(flat_te)
        auc_val = roc_auc_score(test_labels, scores) if len(np.unique(test_labels)) == 2 else 0.5
        baselines[name] = {
            'method': name,
            'auc_roc': round(float(auc_val), 4),
            'ci_95': bootstrap_ci(scores, test_labels),
        }
    result['baselines'] = baselines

    return result


# ─── Strip internal keys before JSON serialization ────────────────

def _strip_internal(d):
    """Remove keys starting with '_' (numpy arrays, internal data)."""
    if isinstance(d, dict):
        return {k: _strip_internal(v) for k, v in d.items() if not k.startswith('_')}
    if isinstance(d, list):
        return [_strip_internal(v) for v in d]
    return d


# ─── Air quality dataset pipeline ────────────────────────────────

def run_aq_dataset(key, ds_info, window_sizes=(6, 12, 24)):
    """Run the full pipeline for one air quality dataset.

    Returns:
        multi_ds_entry: dict for multi_dataset_results.json
        technique_entry: dict for aq_technique_transfer_results.json
    """
    ds = ds_info['loader']()
    name = ds['name']
    data = ds['data']
    sensor_cols = ds['sensor_cols']
    ref_cols = ds['ref_cols']
    n_sensors = len(sensor_cols)
    values = data[sensor_cols].values
    ref_values = data[ref_cols].values

    print(f"\n{'='*60}", flush=True)
    print(f"DATASET: {name} ({n_sensors} sensors, {len(data)} readings)", flush=True)
    print(f"{'='*60}", flush=True)

    scaler = StandardScaler()
    values_sc = scaler.fit_transform(values)

    # ── Technique transfer output ──
    technique = {
        'dataset': name,
        'n_readings': len(data),
        'n_sensors': n_sensors,
        'sensor_cols': sensor_cols,
        'window_sweep_one_tailed': {},
        'temporal_split': {},
    }

    # ── Multi-dataset output (w=6 only) ──
    multi_ds = {
        'dataset': name,
        'n_sensors': n_sensors,
        '_domain': 'air_quality',
    }

    primary_ws = window_sizes[0]

    for ws in window_sizes:
        print(f"\n  --- w={ws}, one-tailed ---", flush=True)
        labels = create_labels(ref_values, ws, percentile=95, two_tailed=False)
        n_win = len(labels)
        af = float(labels.mean())
        print(f"  Windows: {n_win}, Anomalous: {labels.sum()} ({100*af:.1f}%)", flush=True)

        if len(np.unique(labels)) < 2:
            technique['window_sweep_one_tailed'][str(ws)] = {'error': 'no anomalies'}
            continue

        # PBP pipeline
        pbp_res = _run_pbp_pipeline(values, values_sc, labels, n_sensors, ws, sensor_cols)

        # Baselines
        baselines = run_baselines(values_sc, labels, ws, n_sensors)

        ws_entry = {
            'n_windows': n_win,
            'anomaly_frac': round(af, 4),
            'pbp_std': pbp_res['pbp_std'],
            'pbp_pair': pbp_res['pbp_pair'],
            'baselines': baselines,
            'time_std_s': pbp_res['time_std_s'],
            'time_pair_s': pbp_res['time_pair_s'],
        }
        technique['window_sweep_one_tailed'][str(ws)] = ws_entry

        auc_std = pbp_res['pbp_std']['auc_roc']
        auc_pair = pbp_res['pbp_pair']['auc_roc']
        winner = 'std' if auc_std > auc_pair else 'pair'
        gap = abs(auc_std - auc_pair)
        print(f"  PBP std:  {auc_std:.4f}  ({pbp_res['time_std_s']:.0f}s)", flush=True)
        print(f"  PBP pair: {auc_pair:.4f}  ({pbp_res['time_pair_s']:.0f}s)", flush=True)
        print(f"  Winner: {winner} (+{gap:.3f})", flush=True)
        for bn, br in baselines.items():
            print(f"  {bn:<16s}  {br['auc_roc']:.4f}", flush=True)

        # Populate multi_dataset entry from primary window size
        if ws == primary_ws:
            multi_ds['n_windows'] = n_win
            multi_ds['anomaly_fraction'] = round(af, 4)
            multi_ds['pbp_time_s'] = pbp_res['time_std_s']
            multi_ds['pair_time_s'] = pbp_res['time_pair_s']
            multi_ds['pair_ranking'] = pbp_res['pair_ranking']

            # Full-data results in verify_consistency expected format
            full_results = {
                'PBP pair composite': {
                    'method': 'PBP pair composite (5 feats)',
                    'auc_roc': auc_pair,
                    'ci_95': pbp_res['pbp_pair']['ci_95'],
                },
                'PBP composite': {
                    'method': 'PBP composite (5 feats)',
                    'auc_roc': auc_std,
                    'ci_95': pbp_res['pbp_std']['ci_95'],
                },
                'Isolation Forest': baselines['Isolation Forest'],
                'One-Class SVM': baselines['One-Class SVM'],
                'LOF': baselines['LOF'],
            }
            multi_ds['full_data_results'] = full_results

    # Temporal split (on primary window size)
    print(f"\n  --- Temporal split (w={primary_ws}) ---", flush=True)
    ts_result = _run_temporal_split(values, ref_values, n_sensors, primary_ws, sensor_cols)
    technique['temporal_split'] = ts_result

    if 'error' not in ts_result:
        print(f"  Train: anom {100*ts_result['train_anom_frac']:.1f}%", flush=True)
        print(f"  Test:  anom {100*ts_result['test_anom_frac']:.1f}%", flush=True)
        print(f"  TS std:  {ts_result['pbp_std']['auc_roc']:.4f}", flush=True)
        print(f"  TS pair: {ts_result['pbp_pair']['auc_roc']:.4f}", flush=True)

        # Multi-dataset temporal split
        ts_multi = {
            'PBP pair composite': {
                'method': 'PBP pair composite (5 feats)',
                'auc_roc': ts_result['pbp_pair']['auc_roc'],
            },
            'PBP composite': {
                'method': 'PBP composite (5 feats)',
                'auc_roc': ts_result['pbp_std']['auc_roc'],
            },
        }
        if 'baselines' in ts_result:
            for bl_name, bl_data in ts_result['baselines'].items():
                # Map short names to full names
                name_map = {'IF': 'Isolation Forest', 'OC-SVM': 'One-Class SVM', 'LOF': 'LOF'}
                full_name = name_map.get(bl_name, bl_name)
                ts_multi[full_name] = bl_data
        multi_ds['temporal_split_results'] = ts_multi
    else:
        multi_ds['temporal_split_results'] = {'error': ts_result['error']}

    # Two-tailed w=6 (for technique transfer completeness)
    labels_2t = create_labels(ref_values, primary_ws, percentile=95, two_tailed=True)
    if len(np.unique(labels_2t)) == 2:
        pbp_2t = _run_pbp_pipeline(values, values_sc, labels_2t, n_sensors, primary_ws, sensor_cols)
        technique['two_tailed_w6'] = {
            'n_windows': len(labels_2t),
            'anomaly_frac': round(float(labels_2t.mean()), 4),
            'pbp_std': pbp_2t['pbp_std'],
            'pbp_pair': pbp_2t['pbp_pair'],
        }

    return multi_ds, _strip_internal(technique)


# ─── Weather station pipeline ────────────────────────────────────

def run_weather_station(key, ds_info, window_sizes=(6,)):
    """Run the full pipeline for one weather station.

    Returns:
        multi_ds_entry: dict for multi_dataset_results.json
        weather_entry: dict for weather_multi_station_results.json
    """
    ds = ds_info['loader']()
    name = ds['name']
    data = ds['data']
    sensor_cols = ds['sensor_cols']
    ref_cols = ds['ref_cols']
    n_sensors = len(sensor_cols)
    values = data[sensor_cols].values
    ref_values = data[ref_cols].values

    print(f"\n{'='*60}", flush=True)
    print(f"STATION: {name} ({n_sensors} sensors, {len(data)} readings)", flush=True)
    print(f"{'='*60}", flush=True)

    scaler = StandardScaler()
    values_sc = scaler.fit_transform(values)

    ws = window_sizes[0]

    weather = {
        'station': name,
        'code': key,
        'n_readings': len(data),
    }

    multi_ds = {
        'dataset': name,
        'n_sensors': n_sensors,
        '_domain': 'weather',
    }

    # ── One-tailed ──
    print(f"\n  --- w={ws}, one-tailed ---", flush=True)
    labels_1t = create_labels(ref_values, ws, percentile=95, two_tailed=False)
    n_win = len(labels_1t)
    af_1t = float(labels_1t.mean())
    print(f"  Windows: {n_win}, Anomalous: {labels_1t.sum()} ({100*af_1t:.1f}%)", flush=True)

    if len(np.unique(labels_1t)) < 2:
        weather['one_tailed'] = {'error': 'no anomalies'}
    else:
        pbp_1t = _run_pbp_pipeline(values, values_sc, labels_1t, n_sensors, ws, sensor_cols)
        baselines_1t = run_baselines(values_sc, labels_1t, ws, n_sensors)

        weather['one_tailed'] = {
            'anomaly_frac': round(af_1t, 4),
            'pbp_std': pbp_1t['pbp_std'],
            'pbp_pair': pbp_1t['pbp_pair'],
            'baselines': baselines_1t,
        }

        print(f"  PBP std:  {pbp_1t['pbp_std']['auc_roc']:.4f}", flush=True)
        print(f"  PBP pair: {pbp_1t['pbp_pair']['auc_roc']:.4f}", flush=True)
        for bn, br in baselines_1t.items():
            print(f"  {bn:<16s}  {br['auc_roc']:.4f}", flush=True)

        # Multi-dataset entry
        multi_ds['n_windows'] = n_win
        multi_ds['anomaly_fraction'] = round(af_1t, 4)
        multi_ds['pbp_time_s'] = pbp_1t['time_std_s']
        multi_ds['pair_time_s'] = pbp_1t['time_pair_s']
        multi_ds['pair_ranking'] = pbp_1t['pair_ranking']

        multi_ds['full_data_results'] = {
            'PBP pair composite': {
                'method': 'PBP pair composite (5 feats)',
                'auc_roc': pbp_1t['pbp_pair']['auc_roc'],
                'ci_95': pbp_1t['pbp_pair']['ci_95'],
            },
            'PBP composite': {
                'method': 'PBP composite (5 feats)',
                'auc_roc': pbp_1t['pbp_std']['auc_roc'],
                'ci_95': pbp_1t['pbp_std']['ci_95'],
            },
            'Isolation Forest': baselines_1t['Isolation Forest'],
            'One-Class SVM': baselines_1t['One-Class SVM'],
            'LOF': baselines_1t['LOF'],
        }

    # ── Two-tailed ──
    print(f"\n  --- w={ws}, two-tailed ---", flush=True)
    labels_2t = create_labels(ref_values, ws, percentile=95, two_tailed=True)
    af_2t = float(labels_2t.mean())
    print(f"  Windows: {len(labels_2t)}, Anomalous: {labels_2t.sum()} ({100*af_2t:.1f}%)", flush=True)

    if len(np.unique(labels_2t)) < 2:
        weather['two_tailed'] = {'error': 'no anomalies'}
    else:
        pbp_2t = _run_pbp_pipeline(values, values_sc, labels_2t, n_sensors, ws, sensor_cols)
        baselines_2t = run_baselines(values_sc, labels_2t, ws, n_sensors)

        weather['two_tailed'] = {
            'anomaly_frac': round(af_2t, 4),
            'pbp_std': pbp_2t['pbp_std'],
            'pbp_pair': pbp_2t['pbp_pair'],
            'baselines': baselines_2t,
        }

        print(f"  PBP std:  {pbp_2t['pbp_std']['auc_roc']:.4f}", flush=True)
        print(f"  PBP pair: {pbp_2t['pbp_pair']['auc_roc']:.4f}", flush=True)

    # ── Temporal split ──
    print(f"\n  --- Temporal split (w={ws}) ---", flush=True)
    ts_result = _run_temporal_split(values, ref_values, n_sensors, ws, sensor_cols)
    weather['temporal_split'] = ts_result

    if 'error' not in ts_result:
        print(f"  TS std:  {ts_result['pbp_std']['auc_roc']:.4f}", flush=True)
        print(f"  TS pair: {ts_result['pbp_pair']['auc_roc']:.4f}", flush=True)

        multi_ds['temporal_split_results'] = {
            'PBP pair composite': {
                'method': 'PBP pair composite (5 feats)',
                'auc_roc': ts_result['pbp_pair']['auc_roc'],
            },
            'PBP composite': {
                'method': 'PBP composite (5 feats)',
                'auc_roc': ts_result['pbp_std']['auc_roc'],
            },
        }
        if 'baselines' in ts_result:
            for bl_name, bl_data in ts_result['baselines'].items():
                name_map = {'IF': 'Isolation Forest', 'OC-SVM': 'One-Class SVM', 'LOF': 'LOF'}
                multi_ds['temporal_split_results'][name_map.get(bl_name, bl_name)] = bl_data
    else:
        multi_ds['temporal_split_results'] = {'error': ts_result['error']}

    return multi_ds, _strip_internal(weather)


# ─── Cross-environment pair analysis ─────────────────────────────

def compute_cross_env_from_multi(multi_ds_entries):
    """Build cross_env_pair_analysis.json from multi_dataset entries."""
    # Only use air quality datasets (they share pollutant columns)
    aq_entries = [e for e in multi_ds_entries
                  if 'pair_ranking' in e and e.get('pair_ranking')]

    if not aq_entries:
        return {'n_datasets': 0, 'universal_pairs': []}

    dataset_names = [e['dataset'] for e in aq_entries]

    # Collect top-3 pairs per dataset
    pair_presence = defaultdict(lambda: {'datasets': [], 'aucs': []})

    for entry in aq_entries:
        ranking = entry['pair_ranking']
        top3 = ranking[:3]
        for p in top3:
            pair_key = tuple(sorted(p['pair']))
            pair_presence[pair_key]['datasets'].append(entry['dataset'])
            pair_presence[pair_key]['aucs'].append(p['auc'])

    # Find universal pairs (appear in top-3 of >=4 datasets)
    universal = []
    for pair_key, info in pair_presence.items():
        n_ds = len(info['datasets'])
        if n_ds >= 4:
            universal.append({
                'pair': list(pair_key),
                'n_datasets': n_ds,
                'mean_auc': round(float(np.mean(info['aucs'])), 4),
                'std_auc': round(float(np.std(info['aucs'])), 3),
                'datasets': info['datasets'],
            })

    universal.sort(key=lambda x: x['n_datasets'], reverse=True)

    return {
        'n_datasets': len(aq_entries),
        'datasets': dataset_names,
        'universal_pairs': universal,
    }


# ─── JSON output helpers ─────────────────────────────────────────

def _save_json(data, path):
    """Save data to JSON with NaN handling."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f"  Saved: {path}", flush=True)


def _json_default(obj):
    """Handle numpy types and NaN in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Replicate all Nature anomaly experiments')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to run (default: all)')
    parser.add_argument('--quick', action='store_true',
                        help='w=6 only, skip window sweep')
    parser.add_argument('--include-epa', action='store_true',
                        help='Include EPA AQS dataset (requires download)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: ../results/local)')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(SCRIPT_DIR, '..', 'results', 'local')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    window_sizes = (6,) if args.quick else (6, 12, 24)
    registry = _build_dataset_registry(include_epa=args.include_epa)
    targets = args.datasets or list(registry.keys())

    # Validate target names
    invalid = [t for t in targets if t not in registry]
    if invalid:
        print(f"Unknown datasets: {invalid}. Available: {list(registry.keys())}")
        sys.exit(1)

    print(f"PBP backend: {get_backend()}", flush=True)
    print(f"Datasets: {targets}", flush=True)
    print(f"Window sizes: {window_sizes}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"{'='*60}", flush=True)

    # Accumulators for output files
    multi_ds_entries = []
    aq_technique = {}
    weather_results = {}
    errors = {}

    t_start = time.perf_counter()

    for key in targets:
        ds_info = registry[key]
        domain = ds_info['domain']

        try:
            if domain == 'air_quality':
                multi_entry, tech_entry = run_aq_dataset(key, ds_info, window_sizes)
                multi_ds_entries.append(multi_entry)
                aq_technique[key] = tech_entry

            elif domain == 'weather':
                multi_entry, weather_entry = run_weather_station(key, ds_info,
                                                                  window_sizes=(window_sizes[0],))
                multi_ds_entries.append(multi_entry)
                weather_results[key] = weather_entry

        except Exception as e:
            print(f"\n  ERROR on {key}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            errors[key] = str(e)

    t_total = time.perf_counter() - t_start

    # ── Save output files ──

    print(f"\n{'='*60}", flush=True)
    print("SAVING RESULTS", flush=True)
    print(f"{'='*60}", flush=True)

    # 1. multi_dataset_results.json
    _save_json(
        {'datasets': _strip_internal(multi_ds_entries)},
        os.path.join(output_dir, 'multi_dataset_results.json'),
    )

    # 2. aq_technique_transfer_results.json
    if aq_technique:
        _save_json(
            aq_technique,
            os.path.join(output_dir, 'aq_technique_transfer_results.json'),
        )

    # 3. weather_multi_station_results.json
    if weather_results:
        _save_json(
            weather_results,
            os.path.join(output_dir, 'weather_multi_station_results.json'),
        )

    # 4. cross_env_pair_analysis.json (all datasets with pair rankings)
    cross_env = compute_cross_env_from_multi(
        [e for e in multi_ds_entries if 'pair_ranking' in e]
    )
    _save_json(
        cross_env,
        os.path.join(output_dir, 'cross_env_pair_analysis.json'),
    )

    # ── Summary table ──

    print(f"\n{'='*60}", flush=True)
    print("CROSS-DOMAIN COMPARISON MATRIX", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Dataset':<32s} {'Domain':<12s} {'Sens':>4s} {'PBP-std':>8s} "
          f"{'PBP-pair':>9s} {'Win':>5s} {'IF':>8s} {'TS-std':>7s} {'TS-pair':>8s}",
          flush=True)
    print('-' * 100, flush=True)

    def _fmt(v):
        return f"{v:>8.4f}" if isinstance(v, float) else f"{v:>8s}"

    for entry in multi_ds_entries:
        name = entry['dataset']
        ns = entry.get('n_sensors', '?')
        domain = entry.get('_domain', '?')
        full = entry.get('full_data_results', {})
        ts = entry.get('temporal_split_results', {})

        s = full.get('PBP composite', {}).get('auc_roc', '---')
        p = full.get('PBP pair composite', {}).get('auc_roc', '---')
        if_auc = full.get('Isolation Forest', {}).get('auc_roc', '---')
        ts_s = ts.get('PBP composite', {}).get('auc_roc', '---')
        ts_p = ts.get('PBP pair composite', {}).get('auc_roc', '---')
        win = 'std' if isinstance(s, float) and isinstance(p, float) and s > p else 'pair'

        print(f"{name:<32s} {domain:<12s} {ns:>4} {_fmt(s)} {_fmt(p)} "
              f"{win:>5s} {_fmt(if_auc)} {_fmt(ts_s)} {_fmt(ts_p)}", flush=True)

    if errors:
        print(f"\nERRORS ({len(errors)}):", flush=True)
        for key, msg in errors.items():
            print(f"  {key}: {msg[:60]}", flush=True)

    print(f"\nTotal time: {t_total:.0f}s", flush=True)
    print(f"Output dir: {output_dir}", flush=True)


if __name__ == '__main__':
    main()
