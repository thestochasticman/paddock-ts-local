from PaddockTS.PaddockSegmentation._1_presegment import presegment
from PaddockTS.PaddockSegmentation._2_segment import segment
from PaddockTS.Data.download_ds2 import download_ds2
from PaddockTS.query import Query
from os.path import exists


def get_paddocks(
    query: Query,
    min_area_ha: int = 10,
    max_area_ha: int = 1500,
    max_perim_area_ratio: int = 30,
    device='cpu',
    reload=False
):
    if not exists(query.path_ds2) or reload:
        download_ds2(query)
    
    if not exists(query.path_preseg_tif) or reload:
        presegment(query)

    if not exists(query.path_polygons) or reload:
        segment(query, min_area_ha, max_area_ha, max_perim_area_ratio, device=device)

def test():
    from PaddockTS.query import get_example_query

    query = get_example_query()
    get_paddocks(query, device='cpu')

if __name__ == '__main__':
    test()
