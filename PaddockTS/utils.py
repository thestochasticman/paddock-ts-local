from PaddockTS.query import Query
from datetime import date

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
