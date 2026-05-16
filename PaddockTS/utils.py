from PaddockTS.query import Query
from datetime import date
from urllib import request
from urllib.error import URLError

def test_internet(s):
    try:
        request.urlopen('https://www.google.com/', timeout=2)
        return True
    except URLError as error:
        # google.com is not reachable. Check if internet is working or whether google is down'
        return False

def load_user_paddocks(paddocks_filepath: str):
    """Load user-provided paddocks and ensure required columns exist.

    Args:
        paddocks_filepath: Path to a paddocks file (GeoPackage, Shapefile, or GeoJSON).

    Returns:
        GeoDataFrame with 'paddock', 'area_ha', and 'compactness' columns guaranteed.
    """
    import geopandas as gpd
    import numpy as np

    paddocks = gpd.read_file(paddocks_filepath)

    # Ensure 'paddock' column exists
    if 'paddock' not in paddocks.columns:
        paddocks['paddock'] = range(1, len(paddocks) + 1)

    # Compute area_ha if missing (needed by calendar_plot)
    if 'area_ha' not in paddocks.columns:
        metric = paddocks.to_crs(paddocks.estimate_utm_crs())
        paddocks['area_ha'] = metric.geometry.area / 10000

    # Compute compactness if missing
    if 'compactness' not in paddocks.columns:
        metric = paddocks.to_crs(paddocks.estimate_utm_crs())
        paddocks['compactness'] = (4 * np.pi * metric.geometry.area) / (metric.geometry.length ** 2)

    return paddocks


get_example_query = lambda: Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2020, 1, 1),
    end=date(2021, 12, 31),
    stub='RANDOM_PADDOCKTS_QUERY_2'
)

get_example_query2 = lambda: Query.from_lat_lon(
    -35.098087,
    148.929983,
    2,
    date(2025, 6, 1),
    date(2025, 6, 30),
    stub='EXAMPLE_2'
)

get_example_query2 = lambda: Query.from_lat_lon(
    -35.098087,
    148.929983,
    2,
    date(2025, 6, 1),
    date(2025, 6, 30),
    stub='EXAMPLE_3'
)
