from datetime import date
from PaddockTS.query import Query
from PaddockTS.get_outputs import get_outputs

paddocks_fp = "artifacts/Milgadara_paddock-polygons_2024-12-17_12-45-58.json"

q = Query.build_from_paddocks(
    paddocks_filepath=paddocks_fp,
    start=date(2018, 1, 1),
    end=date(2018, 12, 31),
    stub="Migadara_2018",
    label_col="title",
)

get_outputs(
    q,
    paddocks_filepath=paddocks_fp,
    skip_sam=False
)
