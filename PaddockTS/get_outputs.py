from PaddockTS.Environmental.TerrainTiles.download_terrain_tiles import download_terrain
from PaddockTS.Environmental.OzWALD.download_ozwald_daily import download_ozwald_daily
from PaddockTS.Environmental.SLGASoils.download_slgasoils import download_slga_soils
from PaddockTS.Environmental.SILO.download_silo import download_silo

from PaddockTS.IndicesAndVegFrac.veg_frac import compute_fractional_cover
from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
from PaddockTS.PaddockSegmentation2.get_paddocks import get_paddocks

from PaddockTS.Plotting.sentinel2_paddocks_video import sentinel2_video_with_paddocks
from PaddockTS.Plotting.vegfrac_paddocks_video import vegfrac_video_with_paddocks
from PaddockTS.Plotting.terrain_tiles_plot import terrain_tiles_plot
from PaddockTS.Plotting.sentinel2_video import sentinel2_video
from PaddockTS.Plotting.ozwald_plot import ozwald_daily_plot
from PaddockTS.Plotting.vegfrac_video import vegfrac_video
from PaddockTS.Plotting.silo_plot import silo_plot

from PaddockTS.query import Query

def get_outputs(query: Query):
    ds_sentinel2 = download_sentinel2(query)
    sentinel2_video(query, ds_sentinel2)

    download_terrain(query)
    terrain_tiles_plot(query, ds_sentinel2) # this reprojects the terrain tiles onto the sentinel2 geometry before plotting

    download_ozwald_daily(query)
    ozwald_daily_plot(query)

    download_silo(query)
    silo_plot(query)

    download_slga_soils(query)

    ds_vegfrac = compute_fractional_cover(query, ds_sentinel2)
    vegfrac_video(query, ds_vegfrac)

    paddocks = get_paddocks(query, ds_sentinel2)
    sentinel2_video_with_paddocks(query, paddocks, ds_sentinel2)

    vegfrac_video_with_paddocks(query, paddocks, ds_vegfrac, ds_sentinel2)


if __name__ == '__main__':
    from PaddockTS.utils import get_example_query
    query = get_example_query()
    get_outputs(query)
    