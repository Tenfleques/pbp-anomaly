#!/usr/bin/env python3
"""
Verification script for the PBP environmental anomaly detection results.

Checks every number cited in the manuscript against its source CSV/JSON.
Run after any experiment or manuscript change.

Usage:
    python experiments/verify_consistency.py                           # Check precomputed
    python experiments/verify_consistency.py --results-dir results/local  # Check fresh results
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

passed = 0
failed = 0
skipped = 0


def check(description, expected, actual, tol=0.005):
    """Check a single value."""
    global passed, failed
    if isinstance(expected, str) or isinstance(actual, str):
        if str(expected) == str(actual):
            passed += 1
            return True
        else:
            failed += 1
            print(f"  FAIL: {description}: expected={expected}, got={actual}")
            return False

    if abs(expected - actual) <= tol:
        passed += 1
        return True
    else:
        failed += 1
        print(f"  FAIL: {description}: expected={expected}, got={actual}, diff={abs(expected-actual):.6f}")
        return False


def check_multi_dataset_results(results_dir):
    """Verify numbers from multi_dataset_results.json."""
    print("\n--- Multi-dataset results ---")
    path = os.path.join(results_dir, 'multi_dataset_results.json')
    if not os.path.exists(path):
        print("  SKIP: multi_dataset_results.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    datasets = data['datasets']

    # Dataset count
    check("Number of datasets", 5, len(datasets))

    # Per-dataset checks (manuscript Table 1 values)
    expected = {
        'UCI Air Quality (Italy)': {
            'n_windows': 6936,
            'n_sensors': 8,
            'anomaly_fraction': 0.298,
            'pbp_pair_auc_full': 0.890,
            'if_auc_full': 0.855,
            'svm_auc_full': 0.877,
            'pbp_pair_auc_ts': 0.883,
        },
        'Beijing Multi-Site (Dongsi)': {
            'n_sensors': 10,
            'anomaly_fraction': 0.288,
            'pbp_pair_auc_full': 0.902,
            'if_auc_full': 0.927,
            'pbp_pair_auc_ts': 0.906,
        },
        'EPA AQS (LA, 2023)': {
            'n_sensors': 4,
            'anomaly_fraction': 0.293,
            'pbp_pair_auc_full': 0.942,
            'if_auc_full': 0.925,
            'pbp_pair_auc_ts': 0.858,
        },
        'São Paulo (CAMS)': {
            'n_sensors': 6,
            'anomaly_fraction': 0.221,
            'pbp_pair_auc_full': 0.930,
            'if_auc_full': 0.952,
            'pbp_pair_auc_ts': 0.682,
        },
        'Cape Town (CAMS)': {
            'n_sensors': 6,
            'anomaly_fraction': 0.221,
            'pbp_pair_auc_full': 0.967,
            'if_auc_full': 0.983,
            'pbp_pair_auc_ts': 0.971,
        },
    }

    for ds in datasets:
        name = ds['dataset']
        if name not in expected:
            print(f"  WARN: unexpected dataset {name}")
            continue

        exp = expected[name]
        prefix = name[:15]

        if 'n_windows' in exp:
            check(f"{prefix} n_windows", exp['n_windows'], ds['n_windows'], tol=0)
        check(f"{prefix} n_sensors", exp['n_sensors'], ds['n_sensors'], tol=0)
        check(f"{prefix} anomaly_fraction", exp['anomaly_fraction'], ds['anomaly_fraction'])

        pbp_full = ds['full_data_results'].get('PBP pair composite', {}).get('auc_roc', 0)
        check(f"{prefix} PBP pair AUC (full)", exp['pbp_pair_auc_full'], pbp_full)

        if_full = ds['full_data_results'].get('Isolation Forest', {}).get('auc_roc', 0)
        check(f"{prefix} IF AUC (full)", exp['if_auc_full'], if_full)

        if 'svm_auc_full' in exp:
            svm_full = ds['full_data_results'].get('One-Class SVM', {}).get('auc_roc', 0)
            check(f"{prefix} SVM AUC (full)", exp['svm_auc_full'], svm_full)

        pbp_ts = ds['temporal_split_results'].get('PBP pair composite', {}).get('auc_roc', 0)
        check(f"{prefix} PBP pair AUC (temporal)", exp['pbp_pair_auc_ts'], pbp_ts)


def check_cross_env_analysis(results_dir):
    """Verify cross-environment pair analysis results."""
    print("\n--- Cross-environment pair analysis ---")
    path = os.path.join(results_dir, 'cross_env_pair_analysis.json')
    if not os.path.exists(path):
        print("  SKIP: cross_env_pair_analysis.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    check("Number of datasets in cross-env", 5, data['n_datasets'], tol=0)

    # Universal pairs
    universal = data.get('universal_pairs', [])
    check("Number of universal pairs", 4, len(universal), tol=0)

    # CO-O3 should be in all 5 datasets
    co_o3 = [u for u in universal if set(u['pair']) == {'CO', 'O3'}]
    if co_o3:
        check("CO-O3 universality (n_datasets)", 5, co_o3[0]['n_datasets'], tol=0)
        check("CO-O3 mean AUC", 0.869, co_o3[0]['mean_auc'])
    else:
        print("  FAIL: CO-O3 not found in universal pairs")

    # NO2-O3 in 4 datasets
    no2_o3 = [u for u in universal if set(u['pair']) == {'NO2', 'O3'}]
    if no2_o3:
        check("NO2-O3 universality (n_datasets)", 4, no2_o3[0]['n_datasets'], tol=0)


def check_anomaly_rate_sweep(results_dir):
    """Verify anomaly rate sweep results."""
    print("\n--- Anomaly rate sweep ---")
    path = os.path.join(results_dir, 'anomaly_rate_sweep.json')
    if not os.path.exists(path):
        global skipped
        skipped += 1
        print("  SKIP: anomaly_rate_sweep.json not found (Phase 3 not complete)")
        return

    with open(path) as f:
        data = json.load(f)

    # EPA failed (ZIP data on ISTOK NFS, not local), so 4 datasets
    check("Number of datasets in sweep", 4, len(data), tol=0)

    # Each dataset should have sweep entries for percentiles 90-99
    for ds in data:
        n_entries = len(ds['sweep'])
        check(f"{ds['dataset'][:15]} sweep entries", 10, n_entries, tol=0)

        # Anomaly rate should decrease with percentile
        if n_entries >= 2:
            rates = [s['anomaly_rate'] for s in ds['sweep']]
            is_monotone = all(rates[i] >= rates[i+1] for i in range(len(rates)-1))
            check(f"{ds['dataset'][:15]} monotone decreasing rates", True, is_monotone)


def check_seasonal_splits(results_dir):
    """Verify seasonal split results."""
    print("\n--- Seasonal splits ---")
    path = os.path.join(results_dir, 'seasonal_split_results.json')
    if not os.path.exists(path):
        global skipped
        skipped += 1
        print("  SKIP: seasonal_split_results.json not found (Phase 1b not complete)")
        return

    with open(path) as f:
        data = json.load(f)

    check("Seasonal results is dict/list", True, isinstance(data, (dict, list)))


def check_technique_transfer(results_dir):
    """Verify STORY-134 technique transfer results (Table 1 PBP standard + Table 2 weather)."""
    print("\n--- Technique transfer (PBP standard, Table 1) ---")
    path = os.path.join(results_dir, 'aq_technique_transfer_results.json')
    if not os.path.exists(path):
        global skipped
        skipped += 1
        print("  SKIP: aq_technique_transfer_results.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    # Table 1 PBP standard AUC values (full data, w=6, one-tailed)
    expected_std = {
        'UCI': 0.824,
        'Beijing': 0.959,
        'EPA_LA': 0.984,
        'SaoPaulo': 0.964,
        'CapeTown': 0.992,
    }
    expected_pair = {
        'UCI': 0.890,
        'Beijing': 0.902,
        'EPA_LA': 0.942,
        'SaoPaulo': 0.930,
        'CapeTown': 0.967,
    }
    expected_ts_std = {
        'UCI': 0.861,
        'Beijing': 0.932,
        'EPA_LA': 0.934,
        'SaoPaulo': 0.883,
        'CapeTown': 0.973,
    }

    for key in ['UCI', 'Beijing', 'EPA_LA', 'SaoPaulo', 'CapeTown']:
        r = data.get(key, {})
        if 'error' in r:
            print(f"  SKIP: {key} has error: {r['error'][:40]}")
            continue

        ws6 = r.get('window_sweep_one_tailed', {}).get('6', {})
        if 'error' in ws6:
            continue

        s = ws6['pbp_std']['auc_roc']
        p = ws6['pbp_pair']['auc_roc']
        check(f"{key} PBP std AUC (Table 1)", expected_std[key], s)
        check(f"{key} PBP pair AUC (Table 1)", expected_pair[key], p)

        # Temporal split
        ts = r.get('temporal_split', {})
        if 'error' not in ts and 'pbp_std' in ts:
            ts_s = ts['pbp_std']['auc_roc']
            check(f"{key} PBP std AUC (temporal split)", expected_ts_std[key], ts_s)


def check_weather_results(results_dir):
    """Verify weather station results (Table 2)."""
    print("\n--- Weather station results (Table 2) ---")
    path = os.path.join(results_dir, 'weather_multi_station_results.json')
    if not os.path.exists(path):
        global skipped
        skipped += 1
        print("  SKIP: weather_multi_station_results.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    # Table 2 values (one-tailed PBP standard)
    expected = {
        'ORD': {'std_1t': 0.984, 'pair_1t': 0.877, 'ts_std': 0.885, 'anom_1t': 0.269},
        'MIA': {'std_1t': 0.988, 'pair_1t': 0.921, 'ts_std': 0.860, 'anom_1t': 0.307},
        'SFO': {'std_1t': 0.980, 'pair_1t': 0.903, 'ts_std': 0.854, 'anom_1t': 0.292},
        'FAI': {'std_1t': 0.985, 'pair_1t': 0.913, 'ts_std': 0.827, 'anom_1t': 0.258},
    }

    for code, exp in expected.items():
        r = data.get(code, {})
        if 'error' in r:
            print(f"  SKIP: {code} has error")
            continue

        ot = r.get('one_tailed', {})
        s = ot.get('pbp_std', {}).get('auc_roc', 0)
        p = ot.get('pbp_pair', {}).get('auc_roc', 0)
        check(f"{code} PBP std 1T (Table 2)", exp['std_1t'], s)
        check(f"{code} PBP pair 1T (Table 2)", exp['pair_1t'], p)

        ts = r.get('temporal_split', {})
        ts_s = ts.get('pbp_std', {}).get('auc_roc', 0)
        check(f"{code} PBP std TS (Table 2)", exp['ts_std'], ts_s)

    # Standard should beat pair on all 4 stations (one-tailed)
    for code in expected:
        r = data.get(code, {})
        if 'error' in r:
            continue
        s = r['one_tailed']['pbp_std']['auc_roc']
        p = r['one_tailed']['pbp_pair']['auc_roc']
        check(f"{code} std > pair (1T)", True, s > p)


def check_ecod_baseline(results_dir):
    """Verify ECOD baseline results (Table 1)."""
    print("\n--- ECOD baseline ---")
    path = os.path.join(results_dir, 'ecod_baseline_results.json')
    if not os.path.exists(path):
        return
    with open(path) as f:
        data = json.load(f)

    expected_ts = {
        'UCI': 0.709, 'Beijing': 0.756, 'SaoPaulo': 0.644,
        'CapeTown': 0.921, 'EPA_LA': 0.907,
    }
    for key, exp in expected_ts.items():
        actual = data[key]['temporal_split']['auc_roc']
        check(f"ECOD {key} ts AUC", exp, actual)


def check_column_shuffle(results_dir):
    """Verify column-shuffle ablation results."""
    print("\n--- Column shuffle ablation ---")
    path = os.path.join(results_dir, 'column_shuffle_ablation_results.json')
    if not os.path.exists(path):
        return
    with open(path) as f:
        data = json.load(f)

    # Manuscript line 132: "intact superior on 4/5 datasets"
    intact_wins = 0
    for key in ['UCI', 'Beijing', 'EPA_LA', 'SaoPaulo', 'CapeTown']:
        pa = data[key]['protocol_a']
        delta = pa['intact'] - pa['shuffled_mean']
        if delta > 0:
            intact_wins += 1
        check(f"ColShuffle {key} delta small (|d|<0.015)", True, abs(delta) < 0.015)

    check("ColShuffle intact wins >= 4 (Protocol A)", True, intact_wins >= 4)


def check_quantile_sweep(results_dir):
    """Verify quantile discretization sweep (Extended Data Table)."""
    print("\n--- Quantile discretization sweep ---")
    path = os.path.join(results_dir, 'quantile_discretization_sweep.json')
    if not os.path.exists(path):
        return
    with open(path) as f:
        data = json.load(f)

    expected = {
        'SaoPaulo__Q2': (0.883, 0.908), 'SaoPaulo__Q20': (0.956, 0.936),
        'CapeTown__Q2': (0.971, 0.908), 'CapeTown__Q20': (0.989, 0.945),
        'UCI__Q2': (0.871, 0.838), 'UCI__Q3': (0.879, 0.857),
        'Beijing__Q2': (0.949, 0.891), 'Beijing__Q3': (0.931, 0.932),
    }
    for key, (exp_top5, exp_all36) in expected.items():
        if key in data:
            check(f"QSweep {key} top5", exp_top5, data[key]['auc_top5'])
            check(f"QSweep {key} all36", exp_all36, data[key]['auc_all36'])


def check_nonus_weather(results_dir):
    """Verify non-US weather station results (Extended Data Table)."""
    print("\n--- Non-US weather stations ---")
    path = os.path.join(results_dir, 'nonus_weather_results.json')
    if not os.path.exists(path):
        return
    with open(path) as f:
        data = json.load(f)

    expected = {
        'LHR': {'1t': 0.946, '2t': 0.778},
        'NRT': {'1t': 0.928, '2t': 0.693},
    }
    for code, exp in expected.items():
        ot_data = data[code]['one_tailed']
        tt_data = data[code]['two_tailed']
        # Handle both nested dict and flat value formats
        ot = ot_data['pbp_std']['auc_roc'] if isinstance(ot_data.get('pbp_std'), dict) else ot_data.get('pbp_std', 0)
        tt = tt_data['pbp_std']['auc_roc'] if isinstance(tt_data.get('pbp_std'), dict) else tt_data.get('pbp_std', 0)
        check(f"{code} PBP std 1T", exp['1t'], ot)
        check(f"{code} PBP std 2T", exp['2t'], tt)


def check_block_bootstrap_ci(results_dir):
    """Verify block bootstrap CI values in Table 1 and section 2.1."""
    print("\n--- Block bootstrap CIs (Table 1) ---")
    path = os.path.join(results_dir, 'block_bootstrap_ci_results.csv')
    if not os.path.exists(path):
        global skipped
        skipped += 1
        print("  SKIP: block_bootstrap_ci_results.csv not found")
        return

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = {r['dataset']: r for r in reader}

    # Table 1 PBP std AUC [CI] values
    expected = {
        'UCI':      {'auc': 0.861, 'ci_lo': 0.83, 'ci_hi': 0.89, 'diff': -0.017, 'includes_zero': True},
        'Beijing':  {'auc': 0.932, 'ci_lo': 0.92, 'ci_hi': 0.94, 'diff': -0.007, 'includes_zero': True},
        'SaoPaulo': {'auc': 0.883, 'ci_lo': 0.86, 'ci_hi': 0.90, 'diff': -0.012, 'includes_zero': True},
        'CapeTown': {'auc': 0.973, 'ci_lo': 0.96, 'ci_hi': 0.98, 'diff': -0.021, 'includes_zero': False},
        'ORD':      {'auc': 0.885, 'ci_lo': 0.87, 'ci_hi': 0.90, 'diff': -0.018, 'includes_zero': True},
        'MIA':      {'auc': 0.860, 'ci_lo': 0.84, 'ci_hi': 0.88, 'diff': -0.031, 'includes_zero': False},
        'SFO':      {'auc': 0.854, 'ci_lo': 0.83, 'ci_hi': 0.87, 'diff': -0.039, 'includes_zero': False},
        'FAI':      {'auc': 0.827, 'ci_lo': 0.80, 'ci_hi': 0.85, 'diff': -0.084, 'includes_zero': False},
    }

    for key, exp in expected.items():
        if key not in rows:
            print(f"  SKIP: {key} not in bootstrap results")
            continue
        r = rows[key]
        check(f"Bootstrap {key} AUC", exp['auc'], float(r['auc_pbp']))
        check(f"Bootstrap {key} CI lo", exp['ci_lo'], float(r['ci_pbp_lower']), tol=0.01)
        check(f"Bootstrap {key} CI hi", exp['ci_hi'], float(r['ci_pbp_upper']), tol=0.01)
        check(f"Bootstrap {key} diff", exp['diff'], float(r['diff_point']))
        check(f"Bootstrap {key} includes_zero", str(exp['includes_zero']), r['ci_includes_zero'])

    # Manuscript claim: "four of eight comparisons are statistically indistinguishable"
    n_includes_zero = sum(1 for r in rows.values() if r['ci_includes_zero'] == 'True')
    check("4 of 8 CIs include zero", 4, n_includes_zero, tol=0)


def check_interpretability_case_study(results_dir):
    """Verify interpretability case study values in section 2.5."""
    print("\n--- Interpretability case study ---")
    path = os.path.join(results_dir, 'interpretability_case_study_results.json')
    if not os.path.exists(path):
        global skipped
        skipped += 1
        print("  SKIP: interpretability_case_study_results.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    # 5 windows analysed
    check("Case study n_windows", 5, len(data['window_analyses']), tol=0)

    # 28 pairs per window
    w0 = data['window_analyses'][0]
    check("Case study n_pairs", 28, len(w0['pair_scores_ranked']), tol=0)

    # Top pair is NMHC x AH with z=3.94
    top = w0['pair_scores_ranked'][0]
    check("Top pair sensor 1", "PT08.S2(NMHC)", top['pair'][0])
    check("Top pair sensor 2", "AH", top['pair'][1])
    check("Top pair z-score", 3.94, top['z_score'], tol=0.01)

    # z-score range across 28 pairs: 4.83
    zscores = [p['z_score'] for p in w0['pair_scores_ranked']]
    z_range = max(zscores) - min(zscores)
    check("Pair z-score range", 4.83, z_range, tol=0.01)

    # Reference validation: benzene exceeds P95 in 3/5 windows
    ref_vals = data['reference_validation']
    n_benzene_exceeds = sum(
        1 for rv in ref_vals
        if 'PT08.S2(NMHC)' in rv.get('reference_validation', {})
        and rv['reference_validation']['PT08.S2(NMHC)']['exceeds']
    )
    check("Benzene exceeds P95 in 3/5 windows", 3, n_benzene_exceeds, tol=0)

    # Window 112 subtle co-exceedance
    subtle = [s for s in data['specificity_examples'] if s['type'] == 'subtle_coexceedance']
    check("Subtle co-exceedance example exists", True, len(subtle) > 0)
    if subtle:
        check("Window 112 exceedance count", 2, subtle[0]['exceedance_count'], tol=0)
        check("Window 112 pair z-score", 3.94, subtle[0]['z_score'], tol=0.01)

    # Reference P95 threshold for benzene
    check("Benzene P95 threshold", 23.8, data['metadata']['ref_p95_thresholds']['C6H6(GT)'], tol=0.1)

    # Window 1 reference benzene mean = 27.6
    check("Window 1 benzene mean", 27.6, w0['reference_means']['C6H6(GT)'], tol=0.1)


def check_anomaly_transformer(results_dir):
    """Verify Anomaly Transformer results in Table 1 and section 2.1."""
    print("\n--- Anomaly Transformer (Table 1) ---")
    path = os.path.join(results_dir, 'anomaly_transformer_results.json')
    if not os.path.exists(path):
        global skipped
        skipped += 1
        print("  SKIP: anomaly_transformer_results.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    # Table 1 AT values
    check("AT UCI AUC", 0.502, data['UCI']['auc'])
    check("AT Beijing AUC", 0.614, data['Beijing']['auc'], tol=0.005)
    check("AT UCI training time", 83.0, data['UCI']['training_time_s'], tol=5)
    check("AT Beijing training time", 205.0, data['Beijing']['training_time_s'], tol=10)

    # Architecture: 2-layer, 64-dim, 50 epochs
    for ds in ['UCI', 'Beijing']:
        hp = data[ds]['hyperparams']
        check(f"AT {ds} n_layers", 2, hp['n_layers'], tol=0)
        check(f"AT {ds} d_model", 64, hp['d_model'], tol=0)
        check(f"AT {ds} epochs", 50, hp['epochs'], tol=0)


def check_precision_recall(results_dir):
    """Verify precision-recall metrics in Extended Data Table."""
    print("\n--- Precision-recall metrics (ED Table) ---")
    path = os.path.join(results_dir, 'precision_recall_results.csv')
    if not os.path.exists(path):
        global skipped
        skipped += 1
        print("  SKIP: precision_recall_results.csv not found")
        return

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Build lookup: (dataset, method) -> row
    lookup = {(r['dataset'], r['method']): r for r in rows}

    # ED Table values
    expected = [
        ('UCI', 'PBP std', 0.762, 0.435),
        ('UCI', 'Best baseline (IF)', 0.721, 0.458),
        ('UCI', 'Exc. count', 0.613, 0.400),
        ('UCI', 'PBP all-36', 0.679, 0.430),
        ('Beijing', 'PBP std', 0.822, 0.607),
        ('Beijing', 'Best baseline (OC-SVM)', 0.853, 0.569),
        ('Beijing', 'Exc. count', 0.845, 0.595),
        ('SaoPaulo', 'PBP std', 0.435, 0.252),
        ('SaoPaulo', 'Best baseline (IF)', 0.790, 0.479),
        ('SaoPaulo', 'Exc. count', 0.881, 0.659),
        ('CapeTown', 'PBP std', 0.787, 0.552),
        ('CapeTown', 'Best baseline (LOF)', 0.970, 0.895),
        ('CapeTown', 'Exc. count', 0.911, 0.717),
    ]

    for ds, method, exp_prauc, exp_prec90 in expected:
        key = (ds, method)
        if key not in lookup:
            print(f"  SKIP: ({ds}, {method}) not in PR results")
            continue
        r = lookup[key]
        check(f"PR-AUC {ds} {method[:10]}", exp_prauc, float(r['pr_auc']))
        check(f"Prec@90 {ds} {method[:10]}", exp_prec90, float(r['precision_at_recall90']))

    # Discussion claim: PBP precision 0.44-0.61 at 90% recall on UCI and Beijing
    uci_prec = float(lookup[('UCI', 'PBP std')]['precision_at_recall90'])
    bei_prec = float(lookup[('Beijing', 'PBP std')]['precision_at_recall90'])
    check("Discussion: UCI prec@90 ~0.44", True, 0.40 <= uci_prec <= 0.48)
    check("Discussion: Beijing prec@90 ~0.61", True, 0.57 <= bei_prec <= 0.65)


def check_seasonal_detail(results_dir):
    """Verify seasonal split detail values cited in manuscript."""
    print("\n--- Seasonal splits (detail) ---")
    path = os.path.join(results_dir, 'seasonal_split_results.json')
    if not os.path.exists(path):
        return
    with open(path) as f:
        data = json.load(f)

    # Manuscript abstract: SP quarterly CV 0.832 +/- 0.118
    # Manuscript line 282: SP Q2 fold = 0.636, other folds > 0.841
    # Cape Town 0.935 +/- 0.013, Beijing 0.893 +/- 0.008, UCI 0.883 +/- 0.017
    if isinstance(data, dict):
        for ds_key, (exp_mean, exp_std) in [
            ('SaoPaulo', (0.832, 0.118)),
            ('CapeTown', (0.935, 0.013)),
            ('Beijing', (0.893, 0.008)),
            ('UCI', (0.883, 0.017)),
        ]:
            ds = data.get(ds_key, data.get(f'{ds_key}_quarterly', {}))
            if isinstance(ds, dict) and 'quarterly_cv' in ds:
                qcv = ds['quarterly_cv']
                if 'pair_mean' in qcv:
                    check(f"Seasonal {ds_key} pair mean", exp_mean, qcv['pair_mean'])
                    check(f"Seasonal {ds_key} pair std", exp_std, qcv['pair_std'], tol=0.01)


def check_unsupervised_scoring(results_dir):
    """Verify unsupervised scoring values cited in section 2.4 and abstract."""
    print("\n--- Unsupervised scoring (abstract) ---")
    path = os.path.join(results_dir, 'unsupervised_pbp_results.json')
    if not os.path.exists(path):
        return
    with open(path) as f:
        data = json.load(f)

    # Supervised baselines must match Table 1 (block bootstrap CSV)
    boot_path = os.path.join(results_dir, 'block_bootstrap_ci_results.csv')
    if os.path.exists(boot_path):
        import pandas as pd
        boot = pd.read_csv(boot_path)
        for _, r in boot.iterrows():
            ds = r['dataset']
            if ds in data:
                check(f"Unsup sup baseline matches Table 1 ({ds})",
                      r['auc_pbp'], data[ds]['pbp_supervised_top5'])

    # Section 2.4 text values
    check("Beijing exc_multi ~0.930", 0.930, data['Beijing']['exc_count_multi'])
    check("SaoPaulo exc_multi ~0.953", 0.953, data['SaoPaulo']['exc_count_multi'])
    check("CapeTown exc_multi ~0.975", 0.975, data['CapeTown']['exc_count_multi'])
    check("Beijing all36 ~0.864", 0.864, data['Beijing']['pbp_all36'])
    check("SaoPaulo all36 ~0.874", 0.874, data['SaoPaulo']['pbp_all36'])
    check("CapeTown all36 ~0.912", 0.912, data['CapeTown']['pbp_all36'])
    check("UCI all36 ~0.832", 0.832, data['UCI']['pbp_all36'])
    check("UCI exc_multi ~0.782", 0.782, data['UCI']['exc_count_multi'])
    # Abstract: cosine on Cape Town ~0.971 vs supervised ~0.973
    check("Abstract: CapeTown cosine ~0.971", 0.971, data['CapeTown']['pbp_cosine'])
    check("Abstract: CapeTown supervised ~0.973", 0.973, data['CapeTown']['pbp_supervised_top5'])


def main():
    global passed, failed, skipped

    parser = argparse.ArgumentParser(
        description="Verify manuscript numbers against result files."
    )
    parser.add_argument(
        '--results-dir',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'precomputed'),
        help="Directory containing result JSON/CSV files (default: ../results/precomputed)"
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)

    print("=" * 60)
    print("VERIFICATION: PBP Environmental Anomaly Detection")
    print(f"Results directory: {results_dir}")
    print("=" * 60)

    if not os.path.isdir(results_dir):
        print(f"\nERROR: Results directory not found: {results_dir}")
        print("Run experiments first or specify --results-dir.")
        sys.exit(1)

    check_multi_dataset_results(results_dir)
    check_cross_env_analysis(results_dir)
    check_anomaly_rate_sweep(results_dir)
    check_seasonal_splits(results_dir)
    check_seasonal_detail(results_dir)
    check_technique_transfer(results_dir)
    check_weather_results(results_dir)
    check_ecod_baseline(results_dir)
    check_column_shuffle(results_dir)
    check_quantile_sweep(results_dir)
    check_nonus_weather(results_dir)
    check_block_bootstrap_ci(results_dir)
    check_interpretability_case_study(results_dir)
    check_anomaly_transformer(results_dir)
    check_precision_recall(results_dir)
    check_unsupervised_scoring(results_dir)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} PASS, {failed} FAIL, {skipped} SKIP")
    print(f"{'=' * 60}")

    if failed > 0:
        print("\nWARNING: Some checks failed! Investigate before proceeding.")
        sys.exit(1)
    else:
        print("\nAll checks passed.")
        sys.exit(0)


if __name__ == '__main__':
    main()
