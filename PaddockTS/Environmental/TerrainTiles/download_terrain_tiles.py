"""Download a 30 m Copernicus Digital Elevation Model (DEM) tile.

Builds the list of Cloud-Optimised GeoTIFF tile URLs covering the AOI
from `Copernicus DEM 30 m on AWS <https://registry.opendata.aws/copernicus-dem/>`_,
fetches each via ``download_cogs`` (which streams only the bytes that
intersect the AOI), and merges them into a single GeoTIFF written to
``{query.tmp_dir}/Environmental/{query.stub}_terrain.tif``.

Used downstream by :mod:`PaddockTS.Environmental.TerrainTiles.utils`
(slope / TWI / flow accumulation) and
:func:`PaddockTS.Plotting.terrain_tiles_plot`.
"""

from .download_cog import download_cogs
from .check_if_valid_terrain_exists import check_if_valid_terrain_exists
from PaddockTS.query import Query
from datetime import datetime
from itertools import product as cartesian
from math import floor, ceil
from os import makedirs
from os.path import dirname

BASE_URL = 'https://copernicus-dem-30m.s3.amazonaws.com'

def get_cog_urls(bbox):
    """Return the list of Copernicus DEM 30 m tile URLs covering ``bbox``.

    The DEM is tiled at 1° × 1° resolution. This walks every
    integer-degree (lat, lon) cell that intersects ``bbox`` and builds
    the corresponding S3 URL.

    Args:
        bbox: ``[west, south, east, north]`` in EPSG:4326.

    Returns:
        list[str]: One URL per intersecting tile.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    lats = range(floor(lat_min), ceil(lat_max))
    lons = range(floor(lon_min), ceil(lon_max))
    urls = []
    for lat, lon in cartesian(lats, lons):
        ns = f'S{abs(lat):02d}_00' if lat < 0 else f'N{lat:02d}_00'
        ew = f'W{abs(lon):03d}_00' if lon < 0 else f'E{lon:03d}_00'
        tile = f'Copernicus_DSM_COG_10_{ns}_{ew}_DEM'
        urls.append(f'{BASE_URL}/{tile}/{tile}.tif')
    return urls

def download_terrain(query: Query):
    """Download and merge the Copernicus DEM tiles for ``query.bbox``.

    The merged DEM is written to ``query.terrain_path`` (an AOI-keyed
    location, since elevation doesn't depend on the time range). A
    sibling ``.tif._SUCCESS`` marker is touched after the write to mark
    the cache as complete; subsequent calls with the same bbox skip the
    download.
    """
    if check_if_valid_terrain_exists(query.terrain_path):
        return query.terrain_path

    makedirs(dirname(query.terrain_path), exist_ok=True)
    urls = get_cog_urls(query.bbox)
    download_cogs(query.bbox, urls, query.terrain_path)
    # Touch ``<tif>._SUCCESS`` *after* the merge completes; its presence
    # is what the next call uses as the cache-validity check.
    with open(f'{query.terrain_path}._SUCCESS', 'w') as f:
        f.write(datetime.utcnow().isoformat() + 'Z')
    return query.terrain_path


def test():
    from PaddockTS.utils import get_example_query
    download_terrain(get_example_query())


if __name__ == '__main__':
    test()