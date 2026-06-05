# About testing
Each .py script creates a query and runs get_outputs()

## First, test mode 1: user provided AOI and date range
```
python test/test_mode1.py
```

## Second, test mode 2: user provides paddock polygons and date range
This depends on artifacts/*.gpkg as specified in Query.build_from_paddocks() 
```
python test/test_mode2.py # provides .gpkg from hand-drawn polygons somewhere in WA. 
python test/test_mode2_milgadara.py # provides a .json from Agriweb for milgadara
```
## Third, test with user-provided paddocks but run functions individally i.e. don't use get_outputs()
```
python test/slow.py # for this, I used milgadara paddocks .gpkg and 2018-2025.
```

## Results
### jtb June 5 2026 on Mac
test/test_mode1.py -- works

test/test_mode2.py -- cant download S2

test/test_mode2_milgadara.py -- cant download S2

python test/slow.py -- finished running when copy-pasted into python.



