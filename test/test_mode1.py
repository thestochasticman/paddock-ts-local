from datetime import date
from PaddockTS.query import Query
from PaddockTS.get_outputs import get_outputs

query = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],  # [W, S, E, N]
    start=date(2022, 1, 1),
    end=date(2023, 12, 31),
    stub="test_mode1",
)

get_outputs(query, show_log=True)

