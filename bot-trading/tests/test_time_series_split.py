import pytest

from app.validation.time_series_split import SafeTimeSeriesSplit


def test_safe_time_series_split_respects_temporal_order():
    splitter = SafeTimeSeriesSplit(n_splits=3, gap=1, min_train_size=10)

    for train_idx, test_idx in splitter.split(list(range(40))):
        assert max(train_idx) < min(test_idx)


def test_safe_time_series_split_applies_gap():
    splitter = SafeTimeSeriesSplit(n_splits=2, gap=3, min_train_size=10)
    train_idx, test_idx = next(splitter.split(list(range(30))))

    assert min(test_idx) - max(train_idx) == 4


def test_safe_time_series_split_rejects_small_dataset():
    with pytest.raises(ValueError):
        list(SafeTimeSeriesSplit(n_splits=5, gap=1, min_train_size=50).split(list(range(10))))
