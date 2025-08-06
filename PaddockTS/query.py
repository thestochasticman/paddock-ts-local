
from dataclasses_json import dataclass_json
from typing_extensions import Self
from dataclasses import dataclass, field
from datetime import date
from argparse import ArgumentParser
from hashlib import sha256
from typing import Union, Tuple
from PaddockTS.filter import Filter
from os.path import expanduser
from os.path import exists
from os import makedirs
from PaddockTS.utils import *
from os import mkdir
import json


@dataclass_json
@dataclass(frozen=True)
class Query:
    """
    Represents a STAC-query specification, with automatic bounding‐box,
    datetime string, and unique stub generation for caching.

    Attributes:
    (User Defined)
        stub (str)              : Name of the directory where the results of the query are stored
        lat (float)             : Latitude of the area of interest,
        lon (float)             : Longitude of the area of interest,
        buffer (float)          : Buffer in degrees around (lat, lon),
        start_time (date)       : Start date of query (inclusive),
        end_time (date)         : End date of query (inclusive),
        collections (list[str]) : List of STAC collection IDs,
        bands (list[str])       : List of band names to load,
        crs (str)               : Coordinate reference system (default “EPSG:6933”),
        groupby (str)           : ODC groupby key (default “solarday”),
        resolution (int|tuple)  : Spatial resolution in metres (default 10),        
        filter (Filter)         : Expresson to Refine Search                        
        ---------------------------------------------------------------------------
    (Set in __post_init__: __post_init__)
        x (float)               : Same as `lon`,                                           
        y (float)               : Same as `lat`,                                           
        centre (tuple)          : (x, y) pair,                                        
        lon_range (tuple)       : (min_lon, max_lon),                              
        lat_range (tuple)       : (min_lat, max_lat),                              
        datetime (str)          : “YYYY-MM-DD/YYYY-MM-DD” string,                     
        bbox (list)             : [min_lat, min_lon, max_lat, max_lon],                  
    """

    lat         : float             
    lon         : float           
    buffer      : float            
    start_time  : Union[str, date]
    end_time    : Union[str, date]
    collections : list[str]         
    bands       : list[str]      

    filter      : Filter            = Filter.lt("eo:cloud_cover", 10)  
    crs         : str               = 'EPSG:6933'
    groupby     : str               = 'solar_day'
    resolution  : int               = 10
    stub        : Union[str, None]  = None
    out_dir     : str               = expanduser('~/Documents/PaddockTSLocal')
    tmp_dir     : str               = expanduser('~/Downloads/PaddockTSLocal')

    x           : float = field(init=False)
    y           : float = field(init=False)
    centre      : Tuple = field(init=False)
    lon_range   : Tuple = field(init=False)
    lat_range   : Tuple = field(init=False)
    datetime    : str   = field(init=False)
    bbox        : list  = field(init=False)
    
    stub_tmp_dir    : str = field(init=False)
    stub_out_dir    : str = field(init=False)
    path_ds2        : str = field(init=False)
    path_preseg_tif : str = field(init=False)
    path_polygons   : str = field(init=False)

    # Post-init helpers to set derived fields
    set_start_time      = lambda s: object.__setattr__(s, 'start_time', parse_date(s.start_time))
    set_end_time        = lambda s: object.__setattr__(s, 'end_time', parse_date(s.end_time))
    set_x               = lambda s: object.__setattr__(s, 'x', s.lon)
    set_y               = lambda s: object.__setattr__(s, 'y', s.lat)
    set_centre          = lambda s: object.__setattr__(s, 'centre', (s.x, s.y))
    set_lat_range       = lambda s: object.__setattr__(s, 'lat_range', (s.x - s.buffer, s.x + s.buffer))
    set_lon_range       = lambda s: object.__setattr__(s, 'lon_range', (s.y - s.buffer, s.y + s.buffer))
    set_datetime        = lambda s: object.__setattr__(s, 'datetime', f'{str(s.start_time)}/{str(s.end_time)}')
    set_bbox            = lambda s: object.__setattr__(s, 'bbox', [s.lat_range[0], s.lon_range[0], s.lat_range[1], s.lon_range[1]])
    set_stub            = lambda s: object.__setattr__(s, 'stub', s.stub if s.stub is not None else s.get_stub())
    set_stub_tmp_dir    = lambda s: object.__setattr__(s, 'stub_tmp_dir', f"{s.tmp_dir}/{s.stub}")
    set_stub_out_dir    = lambda s: object.__setattr__(s, 'stub_out_dir', f"{s.out_dir}/{s.stub}")
    set_path_ds2        = lambda s: object.__setattr__(s, 'path_ds2', f"{s.stub_tmp_dir}/ds2.pkl")
    set_path_preseg_tif = lambda s: object.__setattr__(s, 'path_preseg_tif', f"{s.stub_tmp_dir}/preseg.tif")
    set_path_polygons   = lambda s: object.__setattr__(s, 'path_polygons', f"{s.stub_tmp_dir}/polygons.gpkg")

    def __post_init__(s: Self) -> None:
        """
        Populate all derived fields after the dataclass is initialized.
        """
        makedirs(s.out_dir, exist_ok=True)
        makedirs(s.tmp_dir, exist_ok=True)

        if isinstance(s.start_time, str):
            object.__setattr__(s, 'start_time', parse_date(s.start_time))
        
        if isinstance(s.end_time, str):
            object.__setattr__(s, 'end_time', parse_date(s.end_time))

        s.set_stub()
        s.set_stub_tmp_dir()
        if not exists(s.stub_tmp_dir): mkdir(s.stub_tmp_dir)
        s.set_stub_out_dir()
        if not exists(s.stub_out_dir): mkdir(s.stub_out_dir)
        s.set_path_ds2()
        s.set_path_preseg_tif()
        s.set_path_polygons()
        s.set_x()
        s.set_y()
        s.set_centre()
        s.set_lon_range()
        s.set_lat_range()
        s.set_datetime()
        s.set_bbox()

    
    def __str__(self: Self) -> str:
        """
        Serialize this Query to a pretty-printed JSON string.

        Returns:
            str: The JSON representation.
        """

        def default(o):
            if isinstance(o, date):
                return o.isoformat()
            if isinstance(o, Filter):
                return o.to_dict()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
        
        init_fields = {
            f.name: getattr(self, f.name)
            for f in self.__dataclass_fields__.values()
            if f.init
        }
        return json.dumps(init_fields, indent=2, default=default)
    

    def get_stub(self: Self) -> str:
        """
        Compute a SHA-256 hash of this Query’s JSON to use as a cache key.

        Returns:
            str: The hex digest stub.
        """
        return sha256(str(self).encode()).hexdigest()

    @classmethod
    def from_cli(cls) -> "Query":
        """
        Parse CLI arguments and construct a Query object.

        Expected flags:
          --lat, --lon, --buffer,
          --start_time, --end_time,
          --collections (one or more), --bands (one or more)

        Returns:
            Query: The populated instance.
        """
        parser = ArgumentParser()
        grp = parser.add_argument_group("query")
        flds = cls.__dataclass_fields__

        grp.add_argument("--stub",        type=str, required=True)
        grp.add_argument("--lat",         type=float, required=True)
        grp.add_argument("--lon",         type=float, required=True)
        grp.add_argument("--buffer",      type=float, required=True)
        grp.add_argument("--start_time",  type=parse_date, required=True)
        grp.add_argument("--end_time",    type=parse_date, required=True)
        grp.add_argument("--collections", nargs='+', required=True)
        grp.add_argument("--bands",       nargs='+', required=True)
        grp.add_argument("--filter",      type=str, required=True)

        args, _ = parser.parse_known_args()

        if args.filter:
            try:
                filter_obj = Filter.from_string(args.filter)
            except Exception as e:
                raise ValueError(f"Invalid --filter value: {e}")
        else:
            filter_obj = Filter.lt("eo:cloud_cover", 10)
        return cls(
            stub=args.stub,
            lat=args.lat,
            lon=args.lon,
            buffer=args.buffer,
            start_time=args.start_time,
            end_time=args.end_time,
            collections=args.collections,
            bands=args.bands,
            filter=filter_obj
        )

def test_query_from_cli():
    import sys
    import pytest
    # Save the original argv so we can restore it later
    original_argv = sys.argv.copy()

    # Build the fake argv list
    sys.argv = [
        "prog",
        "--stub", "test_example_query",
        "--lat", "-33.5040",
        "--lon", "148.4",
        "--buffer", "0.01",
        "--start_time", "2020-01-01",
        "--end_time", "2020-06-01",
        "--collections", "ga_s2am_ard_3", "ga_s2bm_ard_3",
        "--bands", "nbart_blue", "nbart_green", "nbart_red",
        "--filter", "eo:cloud_cover < 10"
    ]

    try:
        # Invoke the CLI parser
        q = Query.from_cli()
    finally:
        # Restore argv so other tests aren’t affected
        sys.argv = original_argv

    # Assertions on parsed values
    assert isinstance(q, Query)
    assert q.lat == pytest.approx(-33.5040)
    assert q.lon == pytest.approx(148.4)
    assert q.buffer == pytest.approx(0.01)
    assert q.start_time == date(2020, 1, 1)
    assert q.end_time == date(2020, 6, 1)
    assert q.collections == ["ga_s2am_ard_3", "ga_s2bm_ard_3"]
    assert q.bands == ["nbart_blue", "nbart_green", "nbart_red"]

    # Check derived fields
    assert q.x == q.lon
    assert q.y == q.lat
    assert q.centre == (q.lon, q.lat)
    assert q.lon_range == (q.y - q.buffer, q.y + q.buffer)
    assert q.lat_range == (q.x - q.buffer, q.x + q.buffer)
    assert q.datetime == "2020-01-01/2020-06-01"
    assert q.bbox == [
        q.lat_range[0],
        q.lon_range[0],
        q.lat_range[1],
        q.lon_range[1]
    ]

    # Stub should be a 64‐character SHA256 hex
    stub = q.get_stub()
    assert isinstance(stub, str) and len(stub) == 64

def get_example_query() -> Query:
    """
    Return a sample Query for testing or demonstration.

    Returns:
        Query: A preset Query covering mid-2020 Sentinel-2 data.
    """
    return Query(
        stub='test_example_query',
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
            'nbart_red_edge_1',
            'nbart_red_edge_2',
            'nbart_red_edge_3',
            'nbart_nir_1',
            'nbart_nir_2',
            'nbart_swir_2',
            'nbart_swir_3'
        ],
    )


if __name__ == '__main__':

    test_query_from_cli()
    q = get_example_query()
    print(q.filter)
    print('passed')