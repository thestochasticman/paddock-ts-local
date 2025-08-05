from PaddockTSLocal.query import Query
from PaddockTSLocal.download_ds2 import download_ds2_from_query
from PaddockTSLocal.ndwi_fourier_geotiff import ds2_to_ndwi_fourier_geotiff
from PaddockTSLocal.fractional_cover import add_fractional_cover_ds2i
from PaddockTSLocal.sam_geo_paddocks import segment
from PaddockTSLocal.paddock_ts import get_paddock_ts

def generate_outputs(stub: str, query: Query):
    download_ds2_from_query(stub, query)
    ds2_to_ndwi_fourier_geotiff(stub)
    add_fractional_cover_ds2i(stub)
    segment(stub)
    get_paddock_ts(stub)
    
def test():
    from PaddockTSLocal.query import get_example_query
    query = get_example_query()

    generate_outputs(query.get_stub(), query)


if __name__ == '__main__':
    test()
