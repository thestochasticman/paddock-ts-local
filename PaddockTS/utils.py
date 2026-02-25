from PaddockTS.query import Query
from datetime import date

get_example_query = lambda: Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2020, 1, 1),
    end=date(2024, 12, 31),
    stub='RANDOM_PADDOCKTS_QUERY'
)