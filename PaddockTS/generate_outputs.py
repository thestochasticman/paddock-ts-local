from PaddockTS.query import Query
from PaddockTS.download_ds2 import download_ds2_from_query
from PaddockTS.ndwi_fourier_geotiff import ds2_to_ndwi_fourier_geotiff
from PaddockTS.fractional_cover import add_fractional_cover_ds2i
from PaddockTS.sam_geo_paddocks import segment
from PaddockTS.paddock_ts import get_paddock_ts
from checkpoint_plots import plot as plot_checkpoint_plots

def generate_outputs(stub: str, query: Query):
    download_ds2_from_query(stub, query)
    ds2_to_ndwi_fourier_geotiff(stub)
    add_fractional_cover_ds2i(stub)
    segment(stub)
    get_paddock_ts(stub)
    plot_checkpoint_plots(stub)
    print('done')
    
def test():
    from PaddockTS.query import get_example_query
    query = get_example_query()

    generate_outputs(query.get_stub(), query)


if __name__ == '__main__':
    test()
