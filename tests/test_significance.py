"""Unit tests: paired significance tests (spec §3.9)."""

import numpy as np
import pandas as pd
import pytest

from armada.eval.significance import exact_sign_flip_pvalue, paired_method_comparison


def test_sign_flip_all_positive_small_n():
    d = np.array([0.1, 0.2, 0.3])
    mean, p = exact_sign_flip_pvalue(d)
    assert mean == pytest.approx(0.2)
    # exact: with n=3 only the all-positive and all-negative assignments are
    # as extreme -> p = 2/8
    assert p == pytest.approx(2 / 8)


def test_sign_flip_zero_diff_is_p_one():
    d = np.array([0.0, 0.0, 0.0])
    _, p = exact_sign_flip_pvalue(d)
    assert p == pytest.approx(1.0)


def test_sign_flip_symmetric_diffs():
    d = np.array([0.2, -0.2])
    _, p = exact_sign_flip_pvalue(d)
    assert p == pytest.approx(1.0)  # no evidence against H0


def test_paired_comparison_pairs_on_seed_order():
    a = [0.9, 0.8, 0.85]
    b = [0.7, 0.75, 0.6]
    r = paired_method_comparison(a, b)
    assert r["n_pairs"] == 3
    assert r["mean_diff"] == pytest.approx(np.mean([0.2, 0.05, 0.25]))
    assert 0.0 < r["p_value"] <= 1.0


def test_paired_comparison_drops_nonfinite():
    a = [0.9, np.nan, 0.85]
    b = [0.7, 0.75, np.nan]
    r = paired_method_comparison(a, b)
    # only index 0 is finite in both -> 1 pair, too few for a test
    assert r["n_pairs"] == 1
    assert np.isnan(r["p_value"])


def test_paired_comparison_needs_equal_length():
    with pytest.raises(ValueError, match="unpaired"):
        paired_method_comparison([0.1, 0.2], [0.1])
