"""Cache-validity check for the SAM-segmented paddocks GeoPackage.

A valid cache requires both the ``.gpkg`` file and a ``.gpkg._SUCCESS``
sentinel sitting beside it (written by :func:`get_paddocks` only after
the GeoPackage write completes). The marker is a *sibling file* rather
than an inside-the-store file because GeoPackages are single-file
SQLite databases, not directories — so the zarr-style "marker inside
the dir" convention doesn't apply.

Mirrors :func:`PaddockTS.Sentinel2.check_if_valid_zarr_exists`.
"""

from os.path import exists


def check_if_valid_paddocks_exists(paddocks_path: str) -> bool:
    """Return True iff a successfully-written paddocks GeoPackage is on disk."""
    return exists(paddocks_path) and exists(f'{paddocks_path}._SUCCESS')


def test_no_file_no_marker():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        return check_if_valid_paddocks_exists(f'{tmpdir}/missing.gpkg') is False


def test_file_without_marker():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        gpkg = f'{tmpdir}/p.gpkg'
        open(gpkg, 'w').close()
        return check_if_valid_paddocks_exists(gpkg) is False


def test_file_with_marker():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        gpkg = f'{tmpdir}/p.gpkg'
        open(gpkg, 'w').close()
        open(f'{gpkg}._SUCCESS', 'w').close()
        return check_if_valid_paddocks_exists(gpkg) is True


def test_marker_without_file():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        open(f'{tmpdir}/p.gpkg._SUCCESS', 'w').close()
        return check_if_valid_paddocks_exists(f'{tmpdir}/p.gpkg') is False


def test():
    return all([
        test_no_file_no_marker(),
        test_file_without_marker(),
        test_file_with_marker(),
        test_marker_without_file(),
    ])


if __name__ == '__main__':
    print(test())
