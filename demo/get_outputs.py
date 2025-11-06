from PaddockTS.get_outputs import get_outputs
from PaddockTS.query import Query
from datetime import date

if __name__ == '__main__':
    query = Query(
        stub='test_example_query',
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
        ],
    )
    get_outputs(query)
        