# About testing
Each .py script creates a query and runs get_outputs()

First, test mode 1: user provided AOI and date range

'
python test/test_mode1.py
'

Second, test mode 2: user provides paddock polygons and date range
This depends on artifacts/*.gpkg as specified in Query.build_from_paddocks() 
'
python test/test_mode2.py
'


