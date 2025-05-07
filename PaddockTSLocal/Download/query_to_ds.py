from PaddockTSLocal.Query import Query
from os.path import join
from os import makedirs
from os import getcwd
import pystac_client
import odc.stac
import pickle


def f(query: Query, path_ds: str):
    
    catalog = pystac_client.Client.open('https://explorer.dea.ga.gov.au/stac')
    odc.stac.configure_rio(
        cloud_defaults=True,
        aws={'aws_unsigned': True},
    )
    filter_expression = {
        "op": "<",
        "args": [{"property": "eo:cloud_cover"}, 10]
    }
    query_results = catalog.search(
        bbox=query.bbox,
        collections=query.collections,
        datetime=query.datetime,
        filter=filter_expression
    )
    items = list(query_results.items())
    ds = odc.stac.load(
        items,
        bands=query.bands,
        crs='epsg:6933',
        resolution=10,
        groupby='solar_day',
        bbox=query.bbox, 
    )
    with open(path_ds, 'wb') as handle:
        pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return ds

def t():
    from datetime import date
    query = Query(
        lat=-33.5040,
        lon=148.4,
        buffer=0.01,
        start_time=date(2020, 1, 1),
        end_time=date(2020, 6, 1),
        collections=['ga_s2am_ard_3', 'ga_s2bm_ard_3'],
        bands=[
            'nbart_blue',
            'nbart_green',
            'nbart_red', 
            'nbart_red_edge_1',
            'nbart_red_edge_2',
            'nbart_red_edge_3',
            'nbart_nir_1',
            'nbart_nir_2',
            'nbart_swir_2',
            'nbart_swir_3'
        ]
    )
    out_dir: str=join(getcwd(), 'Data', 'ds2')
    makedirs(out_dir, exist_ok=True)
    path = join(out_dir, f"{query.get_stub()}.pkl")
    f(query, path)

if __name__ == '__main__':
    t()
