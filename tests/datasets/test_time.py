"""Tests for the TIME task table, windowing and arrow decoding (offline)."""

import inspect
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from benchopt.benchmark import Benchmark


BENCHMARK_DIR = Path(__file__).parents[2]
(Dataset,) = Benchmark(BENCHMARK_DIR).check_dataset_patterns(["TIME"], class_only=True)
time_ds = inspect.getmodule(Dataset)

_YAML = """\
datasets:
  CPHL/15T:
    test_length: 1440  # 15 days
    val_length: 1440
    short:
      prediction_length: 12
    medium:
      prediction_length: 96
    long:
      prediction_length: 288

  Crypto/D:
    test_length: 273
    val_length: 273
    short:
      prediction_length: 30
"""


def _write_arrow(path, targets, freq):
    """Write a tiny HF-style (IPC stream) arrow file."""
    table = pa.table(
        {
            "item_id": [f"item_{i}" for i in range(len(targets))],
            "freq": [freq] * len(targets),
            "target": targets,
        }
    )
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    return str(path)


def test_parse_task_yaml(tmp_path):
    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(_YAML)
    assert time_ds._parse_task_yaml(yaml_path) == {
        "CPHL/15T": dict(
            test_length=1440, terms={"short": 12, "medium": 96, "long": 288}
        ),
        "Crypto/D": dict(test_length=273, terms={"short": 30}),
    }


def test_undefined_term_skips(monkeypatch):
    monkeypatch.setattr(
        time_ds,
        "_tasks_cache",
        {"Crypto/D": dict(test_length=273, terms={"short": 30})},
    )
    ds = Dataset.get_instance(dataset_name="Crypto/D", term="long")
    data = ds.get_data()
    assert set(data) == {"_skip_reason"}
    assert "does not define term 'long'" in data["_skip_reason"]


class TestWindowCutoffs:
    def test_multiple_covers_test_region(self):
        # 1440 / 288 = 5 windows tiling [T - 1440, T) exactly.
        cutoffs = time_ds._window_cutoffs(10_000, 1440, 288)
        assert cutoffs == [8560, 8848, 9136, 9424, 9712]
        assert cutoffs[-1] + 288 == 10_000

    def test_non_multiple_is_start_anchored(self):
        # Crypto/D short: floor(273 / 30) = 9 windows from T - 273; the
        # last 3 steps are never scored (upstream gluonts semantics).
        T = 2842
        cutoffs = time_ds._window_cutoffs(T, 273, 30)
        assert len(cutoffs) == 9
        assert cutoffs[0] == T - 273
        assert np.all(np.diff(cutoffs) == 30)
        assert T - (cutoffs[-1] + 30) == 3

    def test_n_windows_keeps_first(self):
        assert time_ds._window_cutoffs(100, 40, 10, n_windows=2) == [60, 70]

    @pytest.mark.parametrize("n_windows", [0, 5])
    def test_n_windows_out_of_range(self, n_windows):
        with pytest.raises(ValueError, match="n_windows"):
            time_ds._window_cutoffs(100, 40, 10, n_windows=n_windows)

    def test_prediction_length_longer_than_test_length(self):
        with pytest.raises(ValueError, match="exceeds"):
            time_ds._window_cutoffs(100, 20, 30)


def test_build_time_splits():
    ts = np.arange(100, dtype=np.float32).reshape(50, 2)  # (T=50, C=2)
    short = np.zeros((10, 2), dtype=np.float32)  # no history before test
    out = time_ds._build_time_splits([ts, short], test_length=13, prediction_length=4)
    assert out["y_train"] is None
    assert len(out["X_test"]) == 1  # short series dropped
    assert out["cutoff_indexes"] == [[37, 41, 45]]
    np.testing.assert_array_equal(out["X_train"][0], ts[:37])
    assert out["y_test"][0].shape == (3, 4, 2)
    np.testing.assert_array_equal(out["y_test"][0][1], ts[41:45])

    debug = time_ds._build_time_splits(
        [ts], test_length=13, prediction_length=4, debug=True
    )
    assert debug["cutoff_indexes"] == [[37]]


def test_target_to_series_multivariate(tmp_path):
    a = np.arange(8, dtype=np.float32).reshape(2, 4)  # (C=2, T=4)
    b = np.arange(12, dtype=np.float32).reshape(2, 6)  # (C=2, T=6)
    f = _write_arrow(
        tmp_path / "data-00000-of-00001.arrow", [a.tolist(), b.tolist()], "H"
    )
    table = time_ds._read_arrow([f])
    # Slicing (as done for max_series / debug) must respect arrow offsets.
    series = time_ds._target_to_series(table.slice(1, 1).column("target"))
    assert len(series) == 1
    np.testing.assert_array_equal(series[0], b.T)


def test_get_data_end_to_end(monkeypatch, tmp_path):
    a = np.arange(600, dtype=np.float32).reshape(2, 300)
    f = _write_arrow(tmp_path / "data-00000-of-00001.arrow", [a.tolist()], "D")
    monkeypatch.setattr(
        time_ds,
        "_tasks_cache",
        {"Crypto/D": dict(test_length=273, terms={"short": 30})},
    )
    monkeypatch.setattr(Dataset, "_snapshot", lambda self: [f])
    data = Dataset.get_instance(dataset_name="Crypto/D", term="short").get_data()
    assert data["prediction_length"] == 30
    assert data["freq"] == "D"
    assert data["task"] == "forecasting"
    assert data["cutoff_indexes"] == [[27 + 30 * w for w in range(9)]]
    assert data["X_train"][0].shape == (27, 2)
    assert data["y_test"][0].shape == (9, 30, 2)
    assert data["metrics"] == list(time_ds.TIME_METRIC_NAMES)
    assert data["metrics"][0] == "time_mase"  # the objective's ``value``
    assert data["seasonality"] == 1  # gluonts: D -> 1 (from_pandas says 7)


# ``gluonts.time_feature.get_seasonality`` outputs, generated with gluonts
# 0.15.1 (the version upstream TIME pins: ``gluonts~=0.15.1``) on pandas
# 2.3.3. The first 11 are the freqs of the 50 TIME configs; for each, the
# value equals the ``seasonality`` in every upstream TIME-Output
# config.json (Real-TSF/TIME-Output, 1014 configs checked).
GLUONTS_0_15_1_SEASONALITY = {
    "5T": 288, "10T": 144, "15T": 96, "20T": 72, "30T": 48, "H": 24,
    "D": 1, "B": 5, "W": 1, "M": 12, "Q": 4,
    "W-SUN": 1, "W-MON": 1, "MS": 12, "QS-OCT": 4, "Q-DEC": 4, "A": 1,
    "Y": 1, "YE": 1, "AS": 1, "2h": 12, "5min": 288, "S": 3600,
    "10S": 360, "4S": 900, "s": 3600, "h": 24, "ME": 12, "QE": 4,
    "6H": 4, "1D": 1, "3D": 1, "2W": 1, "BM": 1, "BMS": 1, "7T": 1,
    "25T": 1, "1H": 24, "2M": 6, "3MS": 4, "2Q": 2, "L": 1, "ms": 1,
    "U": 1, "N": 1, "BH": 1, "12H": 2, "60T": 24, "90T": 16, "1440T": 1,
}  # fmt: skip


@pytest.mark.parametrize("freq, expected", GLUONTS_0_15_1_SEASONALITY.items())
def test_gluonts_seasonality(freq, expected):
    assert time_ds.gluonts_seasonality(freq) == expected
