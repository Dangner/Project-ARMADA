"""Unit tests: EMBER feature-group slice boundaries (spec §2)."""

import numpy as np
import pytest

from armada.data.groups import (
    GROUPS,
    GROUP_NAMES,
    TOTAL_DIM,
    group_dims,
    group_slices,
    split_vector,
    verify_slice_boundaries,
)


def test_total_dim_is_2381():
    assert TOTAL_DIM == 2381


def test_expected_group_names_in_order():
    assert GROUP_NAMES == (
        "ByteHistogram",
        "ByteEntropyHistogram",
        "StringExtractor",
        "GeneralFileInfo",
        "HeaderFileInfo",
        "SectionInfo",
        "ImportsInfo",
        "ExportsInfo",
        "DataDirectories",
    )


def test_slices_match_ember_feature_version_2():
    expected = {
        "ByteHistogram": (0, 256),
        "ByteEntropyHistogram": (256, 512),
        "StringExtractor": (512, 616),
        "GeneralFileInfo": (616, 626),
        "HeaderFileInfo": (626, 688),
        "SectionInfo": (688, 943),
        "ImportsInfo": (943, 2223),
        "ExportsInfo": (2223, 2351),
        "DataDirectories": (2351, 2381),
    }
    for g in GROUPS:
        assert (g.start, g.stop) == expected[g.name]


def test_slices_are_contiguous_and_sum_to_total():
    rows = verify_slice_boundaries()
    assert sum(r[3] for r in rows) == TOTAL_DIM
    for (name, start, stop, dim), nxt in zip(rows, rows[1:]):
        assert stop == nxt[1], f"gap/overlap between {name} and {nxt[0]}"
        assert dim == stop - start
    assert rows[0][1] == 0
    assert rows[-1][2] == TOTAL_DIM


def test_group_dims_dict():
    dims = group_dims()
    assert dims["ImportsInfo"] == 1280
    assert dims["DataDirectories"] == 30
    assert sum(dims.values()) == TOTAL_DIM


def test_split_vector_shapes_and_values():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, TOTAL_DIM)).astype(np.float32)
    groups = split_vector(X)
    assert set(groups) == set(GROUP_NAMES)
    for g in GROUPS:
        assert groups[g.name].shape == (5, g.dim)
        np.testing.assert_array_equal(groups[g.name], X[:, g.slice])


def test_split_vector_accepts_single_row_and_rejects_bad_shape():
    x = np.zeros(TOTAL_DIM, dtype=np.float32)
    groups = split_vector(x)
    assert groups["GeneralFileInfo"].shape == (1, 10)
    with pytest.raises(ValueError):
        split_vector(np.zeros((3, 100)))
    with pytest.raises(ValueError):
        split_vector(np.zeros(TOTAL_DIM + 1))


def test_group_slices_helper():
    s = group_slices()
    assert s["SectionInfo"] == slice(688, 943)
