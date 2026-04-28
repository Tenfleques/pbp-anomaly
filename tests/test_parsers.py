"""Tests for pbp_anomaly.parsers."""

import os
import numpy as np
import pandas as pd
import pytest
import tempfile

from pbp_anomaly.parsers import load_csv, load_uci_air_quality


class TestLoadCSV:
    def test_basic_load(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        df.to_csv(csv_path, index=False)

        result = load_csv(str(csv_path), ['A', 'B'], name='test')
        assert result['name'] == 'test'
        assert result['sensor_cols'] == ['A', 'B']
        assert result['ref_cols'] == ['A', 'B']
        assert len(result['data']) == 3

    def test_custom_ref_cols(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({'s1': [1, 2], 's2': [3, 4], 'ref': [5, 6]})
        df.to_csv(csv_path, index=False)

        result = load_csv(str(csv_path), ['s1', 's2'], ref_cols=['ref'])
        assert result['ref_cols'] == ['ref']

    def test_missing_column_raises(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({'A': [1]}).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Columns not found"):
            load_csv(str(csv_path), ['A', 'MISSING'])

    def test_drops_nan(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
        df.to_csv(csv_path, index=False)

        result = load_csv(str(csv_path), ['A', 'B'])
        assert len(result['data']) == 1  # Only row 0 has no NaN


class TestLoadUCI:
    def test_loads_if_exists(self):
        path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'publications',
            'articles-2026', 'PBP-environmental-anomaly-nature',
            'experiments', 'data', 'AirQualityUCI.csv'
        )
        if not os.path.exists(path):
            pytest.skip("UCI data not available")

        result = load_uci_air_quality(path)
        assert 'UCI' in result['name']
        assert len(result['sensor_cols']) == 8
        assert len(result['data']) > 6000
