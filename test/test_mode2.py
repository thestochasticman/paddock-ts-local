# jtb: testing running with a manually annotated set of polygons
# geopackage made in QGIS espg4326, one field desgnates paddock name, another field is a random variable to see how it handles it.. 

from datetime import date
from PaddockTS.query import Query
from PaddockTS.get_outputs import get_outputs

# From GeoJSON with custom label column
q = Query.build_from_paddocks(
    
    paddocks_filepath='artifacts/test_paddocks_WA.gpkg',
    start=date(2024, 1, 1),
    end=date(2024, 12, 31),
    label_col='paddock_name',
    stub='WA_test2',
)

# run it:
get_outputs(q)
