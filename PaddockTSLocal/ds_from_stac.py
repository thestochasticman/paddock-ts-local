from PaddockTSLocal.Query import Query
from PaddockTSLocal.Logger import Logger
import pystac_client
import odc.stac
import pickle

def f(query: Query, logger: Logger):
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
    print(query.bbox)
    ds = odc.stac.load(
        items,
        bands=query.bands,
        crs='epsg:6933',
        resolution=10,
        groupby='solar_day',
        bbox=query.bbox, 
    )
    path = logger.get_path_query_dataset(None, query)
    with open(path, 'wb') as handle:
        pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return ds

def t(): f(Query.from_cli(), Logger.from_cli(), stub=4)

if __name__ == '__main__':
    t()
