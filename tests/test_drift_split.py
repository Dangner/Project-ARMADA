"""Unit tests: temporal concept-drift split (spec §2)."""

import numpy as np
import pytest

from armada.data.drift_split import build_drift_split, unique_months


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    months = np.array([f"2017-{m:02d}" for m in (np.arange(n) * 6 // n + 1)])
    y = rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.2, 0.4, 0.4]).astype(np.float32)
    return months, y, rng


def test_unique_months_sorted_chronologically():
    assert unique_months(["2017-12", "2017-01", "2018-01", "2017-06"]) == [
        "2017-01",
        "2017-06",
        "2017-12",
        "2018-01",
    ]


def test_auto_split_windows_are_consecutive_cover_later_months():
    months, y, rng = _toy()
    split = build_drift_split(
        months,
        y,
        {"mode": "auto", "source_fraction": 0.5, "n_target_windows": 2, "val_fraction": 0.1},
        rng=rng,
    )
    assert split.source_months == ("2017-01", "2017-02", "2017-03")
    assert [w.name for w in split.windows] == ["T1", "T2"]
    # 3 remaining months into 2 windows: earlier window takes the remainder.
    assert split.windows[0].months == ("2017-04", "2017-05")
    assert split.windows[1].months == ("2017-06",)

    # labeled eval rows in each window only use that window's months
    for w in split.windows:
        assert set(months[w.labeled_idx]) <= set(w.months)
        assert set(months[w.unlabeled_idx]) <= set(w.months)
        assert np.all(y[w.labeled_idx] != -1)
        assert np.all(y[w.unlabeled_idx] == -1)

    # no row overlap between source train / val / windows
    seen = set()
    for idx in (
        split.source_train_idx,
        split.source_val_idx,
        *[w.labeled_idx for w in split.windows],
        *[w.unlabeled_idx for w in split.windows],
    ):
        assert not (seen & set(idx.tolist()))
        seen |= set(idx.tolist())


def test_explicit_split_respects_given_boundaries():
    months, y, rng = _toy()
    split = build_drift_split(
        months,
        y,
        {
            "mode": "explicit",
            "source_months": ["2017-01", "2017-02"],
            "target_windows": [["2017-03"], ["2017-04", "2017-05"]],
            "val_fraction": 0.1,
        },
        rng=rng,
    )
    assert split.source_months == ("2017-01", "2017-02")
    assert [w.months for w in split.windows] == [("2017-03",), ("2017-04", "2017-05")]


def test_month_counts_logged_for_every_month():
    months, y, rng = _toy()
    split = build_drift_split(
        months,
        y,
        {"mode": "auto", "source_fraction": 0.5, "n_target_windows": 2, "val_fraction": 0.1},
        rng=rng,
    )
    counts = {m["month"]: m for m in split.month_counts}
    assert set(counts) == set(unique_months(months))
    for m in counts.values():
        assert m["n"] == m["malware"] + m["benign"] + m["unlabeled"]


def test_empty_target_domain_raises():
    months, y, rng = _toy()
    with pytest.raises(ValueError, match="target domain"):
        build_drift_split(
            months,
            y,
            {"mode": "auto", "source_fraction": 1.0, "n_target_windows": 2, "val_fraction": 0.1},
            rng=rng,
        )


def test_source_val_fraction_zero():
    months, y, rng = _toy()
    split = build_drift_split(
        months,
        y,
        {"mode": "auto", "source_fraction": 0.5, "n_target_windows": 2, "val_fraction": 0.0},
        rng=rng,
    )
    assert len(split.source_val_idx) == 0
    assert len(split.source_train_idx) > 0


def test_unknown_mode_raises():
    months, y, rng = _toy()
    with pytest.raises(ValueError, match="unknown split.mode"):
        build_drift_split(months, y, {"mode": "bogus"}, rng=rng)
