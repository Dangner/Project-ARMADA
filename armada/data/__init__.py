"""Data package: EMBER schema, loaders, drift split, vectorisation wrapper."""

from .groups import (  # noqa: F401
    GROUPS,
    GROUP_NAMES,
    LOG_TRANSFORM_GROUPS,
    SENTINEL_VALUE,
    TOTAL_DIM,
    group_dims,
    group_slices,
    split_vector,
    verify_slice_boundaries,
)
