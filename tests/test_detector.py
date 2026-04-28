"""Tests for pbp_anomaly.detector."""

import numpy as np
import pandas as pd
import pytest

from pbp_anomaly.detector import (
    AnomalyDetector, detect, pbp_features, create_labels,
    compute_standard_features, compute_pair_features,
    feature_aucs, composite_score, bootstrap_ci, get_backend,
    N_FEATURES,
)


class TestBackend:
    def test_backend_available(self):
        assert get_backend() in ('C', 'Python')


class TestPBPFeatures:
    def test_output_shape(self):
        mat = np.random.randn(6, 3)
        feats = pbp_features(mat)
        assert feats.shape == (6,)

    def test_zero_matrix(self):
        mat = np.zeros((6, 3))
        feats = pbp_features(mat)
        # All-zero input: PBP coefficients are zero, features should be zero
        # (degree may be NaN from create_pbp on degenerate input)
        assert feats[4] == 0  # l1_norm
        assert feats[1] == 0  # monomial_count

    def test_binary_matrix(self):
        mat = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0],
                        [0, 0, 1], [1, 0, 0], [0, 1, 1]], dtype=float)
        feats = pbp_features(mat)
        assert feats[0] > 0  # degree > 0
        assert feats[1] > 0  # monomial_count > 0

    def test_deterministic(self):
        mat = np.random.randn(6, 4)
        f1 = pbp_features(mat)
        f2 = pbp_features(mat)
        np.testing.assert_array_equal(f1, f2)


class TestCreateLabels:
    def test_shape(self):
        values = np.random.randn(100, 4)
        labels = create_labels(values, window_size=6)
        assert labels.shape == (95,)

    def test_binary(self):
        values = np.random.randn(100, 4)
        labels = create_labels(values, window_size=6)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_two_tailed_more_anomalies(self):
        values = np.random.randn(200, 4)
        labels_1t = create_labels(values, two_tailed=False)
        labels_2t = create_labels(values, two_tailed=True)
        assert labels_2t.sum() >= labels_1t.sum()

    def test_extreme_values_detected(self):
        values = np.zeros((20, 2))
        values[10, 0] = 100.0  # extreme high
        labels = create_labels(values, window_size=3, percentile=95)
        # Window containing index 10 should be flagged
        assert labels[8] == 1 or labels[9] == 1 or labels[10] == 1


class TestStandardFeatures:
    def test_output_shape(self):
        values = np.random.randn(50, 3)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        values_sc = sc.fit_transform(values)
        feats = compute_standard_features(values, values_sc, window_size=6)
        # 1 continuous + 5 quantiles = 6 sources * 6 features = 36
        assert feats.shape == (45, 36)


class TestPairFeatures:
    def test_output_shape(self):
        values = np.random.randn(50, 4)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        values_sc = sc.fit_transform(values)
        feats, pairs = compute_pair_features(values, values_sc, n_sensors=4, window_size=6)
        # C(4,2)=6 pairs * 12 features = 72
        assert feats.shape == (45, 72)
        assert len(pairs) == 6

    def test_pair_indices(self):
        _, pairs = compute_pair_features(
            np.random.randn(20, 3), np.random.randn(20, 3), 3, 6)
        assert pairs == [(0, 1), (0, 2), (1, 2)]


class TestFeatureAUCs:
    def test_random_near_half(self):
        features = np.random.randn(100, 5)
        labels = np.random.randint(0, 2, 100)
        aucs = feature_aucs(features, labels)
        assert aucs.shape == (5,)
        # Random features should give AUC near 0.5
        assert all(0.3 < a < 0.7 for a in aucs)

    def test_perfect_feature(self):
        labels = np.array([0]*50 + [1]*50)
        features = np.zeros((100, 1))
        features[50:, 0] = 1.0
        aucs = feature_aucs(features, labels)
        assert aucs[0] == 1.0


class TestCompositeScore:
    def test_output_shape(self):
        features = np.random.randn(50, 10)
        scores = composite_score(features, [0, 3, 7])
        assert scores.shape == (50,)

    def test_normalized(self):
        features = np.random.randn(50, 5)
        scores = composite_score(features, [0, 1, 2])
        # Each normalized feature in [0,1], sum of 3 in [0,3]
        assert scores.min() >= 0
        assert scores.max() <= 3.0 + 1e-10


class TestBootstrapCI:
    def test_output_format(self):
        scores = np.random.randn(100)
        labels = np.random.randint(0, 2, 100)
        ci = bootstrap_ci(scores, labels, n_boot=100)
        assert len(ci) == 2
        assert ci[0] <= ci[1]


class TestAnomalyDetector:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'A': np.random.randn(n),
            'B': np.random.randn(n),
            'C': np.random.randn(n),
        })
        # Inject anomalies
        df.loc[150:160, 'A'] = 5.0
        df.loc[150:160, 'B'] = 4.0
        return df

    def test_standard_mode(self, sample_data):
        detector = AnomalyDetector(window_size=6, mode='standard')
        result = detector.fit_score(sample_data, ['A', 'B', 'C'])
        assert 'standard' in result
        assert 0 <= result['standard']['auc_roc'] <= 1

    def test_pair_mode(self, sample_data):
        detector = AnomalyDetector(window_size=6, mode='pair')
        result = detector.fit_score(sample_data, ['A', 'B', 'C'])
        assert 'pair' in result
        assert 'pair_ranking' in result

    def test_both_mode(self, sample_data):
        detector = AnomalyDetector(window_size=6, mode='both')
        result = detector.fit_score(sample_data, ['A', 'B', 'C'])
        assert 'standard' in result
        assert 'pair' in result

    def test_detect_function(self, sample_data):
        result = detect(sample_data, ['A', 'B', 'C'], mode='standard')
        assert 'scores' in result
        assert 'labels' in result
        assert result['scores'] is not None

    def test_two_tailed(self, sample_data):
        r1 = detect(sample_data, ['A', 'B', 'C'], two_tailed=False)
        r2 = detect(sample_data, ['A', 'B', 'C'], two_tailed=True)
        assert r2['anomaly_fraction'] >= r1['anomaly_fraction']
