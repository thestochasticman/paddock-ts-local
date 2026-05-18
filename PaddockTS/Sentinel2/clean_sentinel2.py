"""Mask cloudy pixels and drop too-cloudy frames from the raw Sentinel-2 cube.

Reads ``query.sentinel2_path`` (raw, fmask included), applies an
fmask-based clear-sky pixel mask (drops the fmask band itself in the
process), then drops scenes whose NaN fraction across all remaining
bands exceeds ``max_nan_fraction``. The cleaned cube is written to
``query.sentinel2_clean_path`` as a Zarr v2 store, cached via the same
``_SUCCESS`` marker contract used by the raw download.
"""

import os
import xarray as xr
from datetime import datetime
from os import makedirs
from xarray import Dataset
from PaddockTS.query import Query
from PaddockTS.Sentinel2.sentinel2 import Sentinel2, defaultsentinel2
from PaddockTS.Sentinel2.check_if_valid_zarr_exists import check_if_valid_zarr_exists


def clean_sentinel2(
    query: Query,
    ds_sentinel2: Dataset | None = None,
    max_nan_fraction: float = 0.5,
    sentinel2: Sentinel2 = defaultsentinel2,
) -> Dataset:
    """Produce a cloud-masked, frame-filtered Sentinel-2 cube.

    Args:
        query: The :class:`PaddockTS.query.Query`. Output is written to
            ``query.sentinel2_clean_path``; raw input is read from
            ``query.sentinel2_path`` if ``ds_sentinel2`` is not provided.
        ds_sentinel2: Optional in-memory raw Sentinel-2 dataset (must still
            include the fmask band). If ``None``, the raw zarr is opened
            (and downloaded first if missing).
        max_nan_fraction: Drop scenes whose NaN fraction (mean over bands,
            x, y) exceeds this threshold. ``0.5`` keeps scenes that are
            at least ~50% clear within the AOI. Default 0.5.
        sentinel2: Config supplying ``cloud_mask_band``, ``fmask_cloud``,
            and ``fmask_shadow``. Defaults to the bundled DEA config.

    Returns:
        xarray.Dataset: The cleaned cube with the fmask band removed and
        only the retained timesteps. Also persisted to
        ``query.sentinel2_clean_path``.
    """
    if check_if_valid_zarr_exists(query.sentinel2_clean_path):
        try:
            return xr.open_zarr(query.sentinel2_clean_path, chunks=None, decode_coords='all')
        except Exception as e:
            print(f'Cache at {query.sentinel2_clean_path} unreadable ({e}); recomputing')

    if ds_sentinel2 is None:
        if not check_if_valid_zarr_exists(query.sentinel2_path):
            from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
            download_sentinel2(query)
        ds = xr.open_zarr(query.sentinel2_path, chunks=None, decode_coords='all')
    else:
        ds = ds_sentinel2

    fmask = ds[sentinel2.cloud_mask_band]
    clear_mask = (fmask != sentinel2.fmask_cloud) & (fmask != sentinel2.fmask_shadow)
    ds = ds.drop_vars(sentinel2.cloud_mask_band).where(clear_mask)

    nan_frac = ds.to_array().isnull().mean(dim=['variable', 'x', 'y'])
    ds = ds.sel(time=nan_frac < max_nan_fraction)

    # Re-pin CRS — drop_vars/where/sel can drop the spatial_ref coord on some
    # xarray versions, and we need ds.rio.crs to survive the zarr round-trip.
    ds = ds.rio.write_crs(sentinel2.crs, inplace=False)
    timestamp = datetime.utcnow().isoformat() + 'Z'
    ds = ds.assign_attrs(cleaned_at=timestamp, max_nan_fraction=max_nan_fraction)

    makedirs(os.path.dirname(query.sentinel2_clean_path), exist_ok=True)
    ds.to_zarr(query.sentinel2_clean_path, mode='w', zarr_format=2)
    with open(f'{query.sentinel2_clean_path}/_SUCCESS', 'w') as f:
        f.write(timestamp)
    return ds


def test_clean_writes_zarr():
    """First call writes the clean zarr at sentinel2_clean_path."""
    import tempfile
    from datetime import date
    from os.path import exists
    from PaddockTS.config import Config

    tmpdir = tempfile.mkdtemp(prefix='paddockts_clean_test_')
    cfg = Config(out_dir=tmpdir, tmp_dir=tmpdir)
    q = Query(
        bbox=[148.36265, -33.52606, 148.38265, -33.50606],
        start=date(2024, 1, 1), end=date(2024, 1, 21),
        stub='clean_write', config=cfg,
    )
    ds = clean_sentinel2(q, max_nan_fraction=0.7)
    return exists(q.sentinel2_clean_path) and ds.time.size > 0


def test_clean_drops_fmask_band():
    """The fmask band must not appear in the cleaned dataset."""
    import tempfile
    from datetime import date
    from PaddockTS.config import Config

    tmpdir = tempfile.mkdtemp(prefix='paddockts_clean_test_')
    cfg = Config(out_dir=tmpdir, tmp_dir=tmpdir)
    q = Query(
        bbox=[148.36265, -33.52606, 148.38265, -33.50606],
        start=date(2024, 1, 1), end=date(2024, 1, 21),
        stub='clean_no_fmask', config=cfg,
    )
    ds = clean_sentinel2(q)
    return defaultsentinel2.cloud_mask_band not in ds.data_vars


def test_clean_uses_cache():
    """Second call with the same query reuses the clean zarr (no rewrite)."""
    import tempfile
    from datetime import date
    from PaddockTS.config import Config

    tmpdir = tempfile.mkdtemp(prefix='paddockts_clean_test_')
    cfg = Config(out_dir=tmpdir, tmp_dir=tmpdir)
    q = Query(
        bbox=[148.36265, -33.52606, 148.38265, -33.50606],
        start=date(2024, 1, 1), end=date(2024, 1, 21),
        stub='clean_reuse', config=cfg,
    )
    clean_sentinel2(q)
    mtime_before = os.path.getmtime(q.sentinel2_clean_path)
    clean_sentinel2(q)
    mtime_after = os.path.getmtime(q.sentinel2_clean_path)
    return mtime_before == mtime_after


def test():
    from PaddockTS.utils import test_internet
    return all([
        test_internet(None),
        test_clean_writes_zarr(),
        test_clean_drops_fmask_band(),
        test_clean_uses_cache(),
    ])


if __name__ == '__main__':
    print(test())
