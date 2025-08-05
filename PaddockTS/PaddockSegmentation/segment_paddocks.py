from PaddockTS.PaddockSegmentation._1_presegment import presegment
from PaddockTS.PaddockSegmentation._2_segment import segment
from PaddockTS.query import Query

def get_paddocks(
    query: Query,
    min_area_ha: int = 10,
    max_area_ha: int = 1500,
    max_perim_area_ratio: int = 30,
    device='cpu'
):
    presegment(query)
    segment(query, min_area_ha, max_area_ha, max_perim_area_ratio, device='cpu')
def test():
    from PaddockTS.query import get_example_query
    
    query = get_example_query()
    segment(query, device='cpu')

if __name__ == '__main__':
    test()
