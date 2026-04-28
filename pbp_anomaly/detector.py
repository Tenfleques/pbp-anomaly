"""
Core anomaly detector using PBP feature extraction.

Supports two modes:
  - standard: full w×s matrix decomposition
  - pair: pairwise sensor sub-matrix decomposition

Automatically selects C backend when available.
"""

import numpy as np
from scipy.stats import entropy
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ─── PBP backend selection ──────────────────────────────────────────────

_USE_C = False
_create_pbp = None

try:
    from pbp.core_c import create_pbp as _create_pbp_c
    _create_pbp = _create_pbp_c
    _USE_C = True
except (ImportError, RuntimeError, OSError):
    pass

if not _USE_C:
    try:
        from pbp.core import create_pbp as _create_pbp_py
        _create_pbp = _create_pbp_py
    except ImportError:
        raise ImportError(
            "pbp package not found. Install it with: "
            "pip install -e ../pbp  (from the pbp-anomaly directory)"
        )


def get_backend():
    """Return which PBP backend is active."""
    return 'C' if _USE_C else 'Python'


# ─── Feature extraction ────────────────────────────────────────────────

def pbp_features(mat):
    """Extract 6 PBP features from a single matrix.

    Features:
        0: degree (max polynomial degree)
        1: monomial_count (nonzero coefficients)
        2: entropy (coefficient distribution entropy)
        3: coeff_std (coefficient standard deviation)
        4: l1_norm (sum of absolute coefficients)
        5: coeff_range (max - min of nonzero coefficients)
    """
    feats = np.zeros(6)
    try:
        result = _create_pbp(mat)
        abs_coeffs = np.abs(result['coeffs'].values)
        nonzero = abs_coeffs[abs_coeffs > 0]

        feats[0] = result['degree'].max()
        feats[1] = len(nonzero)
        if len(nonzero) > 1:
            probs = nonzero / nonzero.sum()
            feats[2] = entropy(probs)
        if len(nonzero) > 0:
            feats[3] = nonzero.std()
        feats[4] = nonzero.sum() if len(nonzero) > 0 else 0
        feats[5] = (nonzero.max() - nonzero.min()) if len(nonzero) > 1 else 0
    except Exception:
        pass
    return feats


FEATURE_NAMES = ['degree', 'monomial_count', 'entropy', 'coeff_std', 'l1_norm', 'coeff_range']
N_FEATURES = 6


# ─── Window and label construction ──────────────────────────────────────

def create_labels(values, window_size=6, percentile=95, two_tailed=False):
    """Create binary anomaly labels for sliding windows.

    Args:
        values: array (n_samples, n_sensors)
        window_size: sliding window length
        percentile: threshold percentile (e.g. 95)
        two_tailed: if True, flag both >Qhigh and <Qlow

    Returns:
        labels: array (n_windows,) of 0/1
    """
    n_windows = (values.shape[0] - window_size) + 1
    high = np.percentile(values, percentile, axis=0)
    labels = np.zeros(n_windows, dtype=int)

    if two_tailed:
        low = np.percentile(values, 100 - percentile, axis=0)
        for i in range(n_windows):
            w = values[i:i + window_size]
            if np.any(w > high) or np.any(w < low):
                labels[i] = 1
    else:
        for i in range(n_windows):
            w = values[i:i + window_size]
            if np.any(w > high):
                labels[i] = 1
    return labels


# ─── Standard (full-matrix) features ────────────────────────────────────

def compute_standard_features(values, values_scaled, window_size=6,
                              quantiles=(25, 50, 75, 90, 95)):
    """Compute PBP features on full w×s windows at multiple binarization thresholds.

    Returns array of shape (n_windows, (1 + len(quantiles)) * 6).
    First 6 features are from continuous (scaled) data; remaining are from
    binarized data at each quantile threshold.
    """
    thresholds = [np.percentile(values, q, axis=0) for q in quantiles]
    n_windows = (values.shape[0] - window_size) + 1
    n_sources = 1 + len(thresholds)
    features = np.zeros((n_windows, n_sources * N_FEATURES))

    for i in range(n_windows):
        mat_sc = values_scaled[i:i + window_size]
        features[i, :N_FEATURES] = pbp_features(mat_sc)

        mat_raw = values[i:i + window_size]
        for t_idx, thresh in enumerate(thresholds):
            binary = (mat_raw > thresh).astype(float)
            offset = N_FEATURES + t_idx * N_FEATURES
            features[i, offset:offset + N_FEATURES] = pbp_features(binary)

        if (i + 1) % 3000 == 0:
            print(f"    std: {i + 1}/{n_windows}", flush=True)

    return features


# ─── Pair (pairwise sensor) features ────────────────────────────────────

def compute_pair_features(values, values_scaled, n_sensors, window_size=6,
                          binarize_quantile=90):
    """Compute PBP features on w×2 sensor-pair sub-matrices.

    Returns:
        features: array (n_windows, n_pairs * 12)
        pairs: list of (i, j) sensor index tuples
    """
    pairs = list(combinations(range(n_sensors), 2))
    n_pairs = len(pairs)
    n_windows = (values.shape[0] - window_size) + 1
    q_thresh = np.percentile(values, binarize_quantile, axis=0)
    features = np.zeros((n_windows, n_pairs * 12))

    for i in range(n_windows):
        for p_idx, (s1, s2) in enumerate(pairs):
            mat_sc = values_scaled[i:i + window_size, [s1, s2]]
            mat_raw = values[i:i + window_size, [s1, s2]]
            offset = p_idx * 12
            features[i, offset:offset + N_FEATURES] = pbp_features(mat_sc)
            binary = (mat_raw > q_thresh[[s1, s2]]).astype(float)
            features[i, offset + N_FEATURES:offset + 12] = pbp_features(binary)

        if (i + 1) % 3000 == 0:
            print(f"    pair: {i + 1}/{n_windows}", flush=True)

    return features, pairs


# ─── Scoring ────────────────────────────────────────────────────────────

def feature_aucs(features, labels):
    """Compute per-feature AUC-ROC against labels."""
    aucs = np.zeros(features.shape[1])
    for j in range(features.shape[1]):
        col = features[:, j]
        if np.std(col) > 1e-10 and len(np.unique(labels)) == 2:
            try:
                aucs[j] = roc_auc_score(labels, col)
            except ValueError:
                aucs[j] = 0.5
        else:
            aucs[j] = 0.5
    return aucs


def composite_score(features, top_indices):
    """Normalized sum of top-K features."""
    normed = np.zeros((features.shape[0], len(top_indices)))
    for k, j in enumerate(top_indices):
        col = features[:, j]
        lo, hi = col.min(), col.max()
        if hi > lo:
            normed[:, k] = (col - lo) / (hi - lo)
    return normed.sum(axis=1)


def bootstrap_ci(scores, labels, n_boot=2000, seed=42, alpha=0.05):
    """Bootstrap confidence interval for AUC-ROC."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(labels[idx], scores[idx]))
    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100
    return [round(float(x), 4) for x in np.percentile(boot_aucs, [lo, hi])]


# ─── High-level detector ───────────────────────────────────────────────

class AnomalyDetector:
    """Training-free anomaly detector using PBP feature extraction.

    Args:
        window_size: sliding window length (default 6)
        mode: 'standard', 'pair', or 'both' (default 'standard')
        percentile: anomaly threshold percentile (default 95)
        two_tailed: flag both high and low extremes (default False)
        top_k: number of features for composite score (default 5)
    """

    def __init__(self, window_size=6, mode='standard', percentile=95,
                 two_tailed=False, top_k=5):
        self.window_size = window_size
        self.mode = mode
        self.percentile = percentile
        self.two_tailed = two_tailed
        self.top_k = top_k

        self.features_std_ = None
        self.features_pair_ = None
        self.labels_ = None
        self.scores_ = None
        self.pair_ranking_ = None

    def fit_score(self, data, sensor_cols, ref_cols=None):
        """Extract features and compute anomaly scores.

        Args:
            data: DataFrame with sensor readings
            sensor_cols: list of column names for sensor channels
            ref_cols: columns for ground truth labels (default: same as sensor_cols)

        Returns:
            dict with scores, labels, AUC, pair_ranking
        """
        if ref_cols is None:
            ref_cols = sensor_cols

        values = data[sensor_cols].values
        n_sensors = len(sensor_cols)
        ws = self.window_size

        scaler = StandardScaler()
        values_sc = scaler.fit_transform(values)

        # Labels
        self.labels_ = create_labels(
            data[ref_cols].values, ws, self.percentile, self.two_tailed
        )
        n_win = len(self.labels_)

        result = {
            'n_windows': n_win,
            'n_sensors': n_sensors,
            'anomaly_fraction': round(float(self.labels_.mean()), 4),
            'backend': get_backend(),
        }

        if self.mode in ('standard', 'both'):
            self.features_std_ = compute_standard_features(values, values_sc, ws)
            aucs = feature_aucs(self.features_std_, self.labels_)
            top_k = list(np.argsort(aucs)[::-1][:self.top_k])
            scores = composite_score(self.features_std_, top_k)

            auc = roc_auc_score(self.labels_, scores) if len(np.unique(self.labels_)) == 2 else 0.5
            result['standard'] = {
                'auc_roc': round(float(auc), 4),
                'ci_95': bootstrap_ci(scores, self.labels_),
            }
            self.scores_ = scores

        if self.mode in ('pair', 'both'):
            self.features_pair_, pairs = compute_pair_features(
                values, values_sc, n_sensors, ws
            )
            aucs = feature_aucs(self.features_pair_, self.labels_)
            top_k = list(np.argsort(aucs)[::-1][:self.top_k])
            scores = composite_score(self.features_pair_, top_k)

            auc = roc_auc_score(self.labels_, scores) if len(np.unique(self.labels_)) == 2 else 0.5
            result['pair'] = {
                'auc_roc': round(float(auc), 4),
                'ci_95': bootstrap_ci(scores, self.labels_),
            }

            # Pair ranking
            pair_ranking = []
            for p_idx, (s1, s2) in enumerate(pairs):
                pair_aucs = aucs[p_idx * 12:(p_idx + 1) * 12]
                pair_ranking.append({
                    'pair': f"{sensor_cols[s1]}–{sensor_cols[s2]}",
                    'auc': round(float(pair_aucs.max()), 4),
                })
            pair_ranking.sort(key=lambda x: x['auc'], reverse=True)
            self.pair_ranking_ = pair_ranking
            result['pair_ranking'] = pair_ranking

            if self.mode == 'pair':
                self.scores_ = scores

        return result


def detect(data, sensor_cols, ref_cols=None, window_size=6, mode='standard',
           percentile=95, two_tailed=False, top_k=5):
    """One-shot anomaly detection on a DataFrame.

    Args:
        data: DataFrame with sensor readings
        sensor_cols: list of sensor column names
        ref_cols: reference columns for labels (default: sensor_cols)
        window_size: sliding window length
        mode: 'standard', 'pair', or 'both'
        percentile: threshold percentile
        two_tailed: flag both extremes
        top_k: features for composite score

    Returns:
        dict with AUC, scores, labels, pair_ranking
    """
    detector = AnomalyDetector(
        window_size=window_size, mode=mode, percentile=percentile,
        two_tailed=two_tailed, top_k=top_k,
    )
    result = detector.fit_score(data, sensor_cols, ref_cols)
    result['scores'] = detector.scores_
    result['labels'] = detector.labels_
    return result
