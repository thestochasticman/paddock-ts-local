"""Cache-validity check for downloaded Sentinel-2 zarrs.

A valid cache requires both the zarr directory and a ``_SUCCESS`` marker
file *inside* it (written by :func:`download_sentinel2` only after the
zarr write completes). Without the marker the zarr is treated as a
partial / interrupted write and re-downloaded.

Keeping the marker inside the zarr dir means each zarr carries its own
completion state — important when multiple zarrs share the same parent
directory (e.g. ``sentinel2.zarr`` and ``indices.zarr`` under one query dir).
"""

from os.path import exists


def check_if_valid_zarr_exists(zarr_path: str) -> bool:
    """Return True iff a successfully-written zarr is on disk at ``zarr_path``.

    Checks both:
    - the zarr directory itself exists, and
    - a ``_SUCCESS`` marker file exists inside it.
    """
    return exists(zarr_path) and exists(f'{zarr_path}/_SUCCESS')


def test_no_zarr_no_marker():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        return check_if_valid_zarr_exists(f'{tmpdir}/missing.zarr') is False


def test_zarr_without_marker():
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f'{tmpdir}/data.zarr'
        os.makedirs(zarr_path)
        return check_if_valid_zarr_exists(zarr_path) is False


def test_zarr_with_marker():
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f'{tmpdir}/data.zarr'
        os.makedirs(zarr_path)
        open(f'{zarr_path}/_SUCCESS', 'w').close()
        return check_if_valid_zarr_exists(zarr_path) is True


def test_marker_without_zarr():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        return check_if_valid_zarr_exists(f'{tmpdir}/missing.zarr') is False


def test():
    return all([
        test_no_zarr_no_marker(),
        test_zarr_without_marker(),
        test_zarr_with_marker(),
        test_marker_without_zarr(),
    ])


if __name__ == '__main__':
    print(test())
