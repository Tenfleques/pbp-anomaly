"""
CLI for pbp-anomaly.

Usage:
    pbp-anomaly detect data.csv --sensors CO,NO2,O3 --mode standard
    pbp-anomaly detect data.csv --sensors CO,NO2,O3 --mode both --window 12
    pbp-anomaly info
"""

import argparse
import json
import sys

from pbp_anomaly.detector import AnomalyDetector, get_backend


def cmd_detect(args):
    from pbp_anomaly.parsers import load_csv

    sensor_cols = [s.strip() for s in args.sensors.split(',')]
    ref_cols = [s.strip() for s in args.ref.split(',')] if args.ref else None

    ds = load_csv(args.csv, sensor_cols, ref_cols, name=args.csv)

    detector = AnomalyDetector(
        window_size=args.window,
        mode=args.mode,
        percentile=args.percentile,
        two_tailed=args.two_tailed,
        top_k=args.top_k,
    )

    print(f"Dataset: {ds['name']}")
    print(f"Readings: {len(ds['data'])}, Sensors: {len(sensor_cols)}")
    print(f"Backend: {get_backend()}")
    print(f"Mode: {args.mode}, Window: {args.window}, Percentile: {args.percentile}")
    print()

    result = detector.fit_score(ds['data'], ds['sensor_cols'], ds['ref_cols'])

    print(f"Windows: {result['n_windows']}")
    print(f"Anomaly rate: {100 * result['anomaly_fraction']:.1f}%")
    print()

    if 'standard' in result:
        r = result['standard']
        print(f"PBP standard:  AUC = {r['auc_roc']:.4f}  CI = {r['ci_95']}")
    if 'pair' in result:
        r = result['pair']
        print(f"PBP pair:      AUC = {r['auc_roc']:.4f}  CI = {r['ci_95']}")
        if 'pair_ranking' in result:
            print(f"\nTop 5 sensor pairs:")
            for p in result['pair_ranking'][:5]:
                print(f"  {p['pair']:<20s} AUC = {p['auc']:.4f}")

    if args.output:
        # Remove non-serializable arrays
        out = {k: v for k, v in result.items() if k not in ('scores', 'labels')}
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


def cmd_info(args):
    from pbp_anomaly import __version__
    print(f"pbp-anomaly v{__version__}")
    print(f"PBP backend: {get_backend()}")
    print(f"Features per window: 6 (degree, monomial_count, entropy, coeff_std, l1_norm, coeff_range)")
    print(f"Modes: standard (full-matrix), pair (sensor-pair decomposition), both")


def main():
    parser = argparse.ArgumentParser(
        prog='pbp-anomaly',
        description='Training-free anomaly detection via PBP decomposition',
    )
    sub = parser.add_subparsers(dest='command')

    # detect
    p_detect = sub.add_parser('detect', help='Run anomaly detection on a CSV file')
    p_detect.add_argument('csv', help='Path to CSV file')
    p_detect.add_argument('--sensors', required=True, help='Comma-separated sensor column names')
    p_detect.add_argument('--ref', default=None, help='Reference columns for labels (default: same as sensors)')
    p_detect.add_argument('--mode', default='standard', choices=['standard', 'pair', 'both'])
    p_detect.add_argument('--window', type=int, default=6, help='Window size (default: 6)')
    p_detect.add_argument('--percentile', type=int, default=95, help='Threshold percentile (default: 95)')
    p_detect.add_argument('--two-tailed', action='store_true', help='Flag both high and low extremes')
    p_detect.add_argument('--top-k', type=int, default=5, help='Top-K features for composite (default: 5)')
    p_detect.add_argument('--output', '-o', help='Save results to JSON file')

    # info
    sub.add_parser('info', help='Show library info')

    args = parser.parse_args()
    if args.command == 'detect':
        cmd_detect(args)
    elif args.command == 'info':
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
