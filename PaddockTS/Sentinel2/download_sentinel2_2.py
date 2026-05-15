import dask.distributed
import folium
import folium.plugins
import geopandas as gpd
import odc.ui
import shapely.geometry
import yaml
from branca.element import Figure
from IPython.display import HTML, display
from odc.algo import to_rgba
from pystac_client import Client
from odc.stac import configure_rio, stac_load

from PaddockTS.query import Query
from odc.stac import configure_rio, stac_load

from PaddockTS.Sentinel2.sentinel2 import Sentinel2
from PaddockTS.Sentinel2.download_sentinel2 import _s3_to_https


cfg = """---
sentinel-s2-l2a-cogs:
  assets:
    '*':
      data_type: uint16
      nodata: 0
      unit: '1'
    SCL:
      data_type: uint8
      nodata: 0
      unit: '1'
    visual:
      data_type: uint8
      nodata: 0
      unit: '1'
  aliases:  # Alias -> Canonical Name
    red: B04
    green: B03
    blue: B02
"*":
  warnings: ignore # Disable warnings about duplicate common names
"""
cfg = yaml.load(cfg, Loader=yaml.SafeLoader)

s = Sentinel2()
def download_sentinel2(query: Query):

    client = dask.distributed.Client()
    configure_rio(cloud_defaults=True, aws={"aws_unsigned": True}, client=client)
    display(client)

    catalog = Client.open('https://explorer.dea.ga.gov.au/stac')
    odc_query = catalog.search(
        collections=s.collections,
        bbox=query.bbox,
        datetime=f'{query.start}/{query.end}',
        filter=s.cloud_cover_filter
    )
    items = list(odc_query.items())

    ds = stac_load(
        items,
        bands=s.bands,
        resolution=s.resolution,
        chunks={},  # <-- use Dask
        groupby="solar_day",
        crs=s.crs,
        patch_url=_s3_to_https
    )
    ds.compute()
    print(ds)


if __name__ == '__main__':
    from PaddockTS.utils import get_example_query
    download_sentinel2(get_example_query())
