from PaddockTS.get_outputs import get_outputs
import sys

if __name__ == '__main__':
    from PaddockTS.utils import get_example_query
    from PaddockTS.config import Config
    from PaddockTS.query import Query
    from datetime import date
    
    from datetime import date
    from PaddockTS.query import Query
    from PaddockTS.get_outputs import get_outputs
    
    # From GeoJSON with custom label column
    q = Query.build_from_paddocks(
        paddocks_filepath='artifacts/test_paddocks_WA.gpkg',
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        label_col='paddock_name',
        stub='WA_test1',
    )
    
    # run it:
    get_outputs(q, show_log=True)