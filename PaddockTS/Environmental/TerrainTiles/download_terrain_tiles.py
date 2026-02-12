from .download_cog import download_cogs
from PaddockTS.query import Query
from itertools import product as cartesian
from math import floor, ceil
from os import makedirs

get_filename = lambda q: f'{q.tmp_dir}/Environmental/{q.stub}_terrain.tif'

BASE_URL = 'https://copernicus-dem-30m.s3.amazonaws.com'

def get_cog_urls(bbox):
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
    makedirs(f'{query.tmp_dir}/Environmental', exist_ok=True)
    urls = get_cog_urls(query.bbox)
    filename = get_filename(query)
    download_cogs(query.bbox, urls, filename)


def test():
    from PaddockTS.utils import get_example_query
    download_terrain(get_example_query())


if __name__ == '__main__':
    test()