"""
pbp-anomaly: Training-free anomaly detection via pseudo-Boolean polynomial decomposition.

Detects anomalies in multivariate sensor streams by extracting polynomial features
from binarized sliding windows. No training data required.

Usage:
    from pbp_anomaly import detect, AnomalyDetector

    detector = AnomalyDetector(window_size=6, thresholds='auto')
    scores = detector.fit_score(dataframe, sensor_cols=['CO', 'NO2', 'O3'])

    # Or one-shot:
    result = detect(dataframe, sensor_cols=['CO', 'NO2', 'O3'])
"""

from pbp_anomaly.detector import AnomalyDetector, detect

__all__ = ['AnomalyDetector', 'detect']
__version__ = '0.2.0'
