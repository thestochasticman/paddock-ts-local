from PaddockTS.query import Query
from datetime import date
import sys
from PaddockTS.filter import Filter


### You can use the Query.cli method to get query from command line

sys.argv = [
    "prog",
    "--lat", "-33.5040",
    "--lon", "148.4",
    "--buffer", "0.01",
    "--start_time", "2020-01-01",
    "--end_time", "2020-06-01",
    "--collections", "ga_s2am_ard_3", "ga_s2bm_ard_3",
    "--bands", "nbart_blue", "nbart_green", "nbart_red",
    "--filter", "eo:cloud_cover < 10"
]

### The above args are the equivalent of putting args while running script
### in the following way
### python demo/query.py 
### --lat -33.5040 --lon 148.04 --buffer 0.01 \
### --start_time  2020-01-01
### --end_time    2020-06-01
### collections ga_s2am_ard_3, ga_s2bm_ard_3
### bands nbartr_blue, nbart_green, nbart_red
### filter  "eo:cloud_cover<10"

q1 = Query.from_cli()


## You can instantiate the same quary programatically

q2 = Query(
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
    ],
    filter=Filter.lt('eo:cloud_cover', 10)
)

