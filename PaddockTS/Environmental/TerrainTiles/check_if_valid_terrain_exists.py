"""Cache-validity check for the downloaded Copernicus DEM TIFF.

A valid cache requires both the ``.tif`` file and a ``.tif._SUCCESS``
sentinel sitting beside it (written by :func:`download_terrain` only
after the GeoTIFF merge completes). The marker is a *sibling file*
because GeoTIFFs are single files, not directory stores.

Mirrors :func:`PaddockTS.PaddockSegmentation.check_if_valid_paddocks_exists`.
"""

from os.path import exists


def check_if_valid_terrain_exists(terrain_path: str) -> bool:
    """Return True iff a successfully-downloaded terrain TIFF is on disk."""
    return exists(terrain_path) and exists(f'{terrain_path}._SUCCESS')


def test_no_file_no_marker():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        return check_if_valid_terrain_exists(f'{tmpdir}/missing.tif') is False


def test_file_without_marker():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tif = f'{tmpdir}/t.tif'
        open(tif, 'w').close()
        return check_if_valid_terrain_exists(tif) is False


def test_file_with_marker():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tif = f'{tmpdir}/t.tif'
        open(tif, 'w').close()
        open(f'{tif}._SUCCESS', 'w').close()
        return check_if_valid_terrain_exists(tif) is True


def test():
    return all([
        test_no_file_no_marker(),
        test_file_without_marker(),
        test_file_with_marker(),
    ])


if __name__ == '__main__':
    print(test())
