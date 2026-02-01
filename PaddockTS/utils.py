from PaddockTS.query import Query
from datetime import date

lambda get_example_query:  Query(
    bbox=[148.37, -33.52, 148.38, -33.51],  # ~1km × 1km
    start=date(2023, 1, 1),
    end=date(2023, 12, 31),
)