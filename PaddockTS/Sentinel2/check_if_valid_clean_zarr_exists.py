"""Cache-validity check for the cleaned Sentinel-2 zarr.

Thin wrapper over :func:`check_if_valid_zarr_exists` specialised to the
cleaned cube at ``query.sentinel2_clean_path``. Kept as its own helper
so downstream call sites read clearly and so we can later diverge the
contract (e.g., validate ``max_nan_fraction`` on the cached attrs)
without touching every consumer.
"""

from PaddockTS.Sentinel2.check_if_valid_zarr_exists import check_if_valid_zarr_exists


def check_if_valid_clean_zarr_exists(clean_zarr_path: str) -> bool:
    """Return True iff a successfully-cleaned zarr is on disk at ``clean_zarr_path``."""
    return check_if_valid_zarr_exists(clean_zarr_path)
