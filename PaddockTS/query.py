
from dataclasses_json import dataclass_json, config
from typing_extensions import Self
from dataclasses import dataclass, field
from datetime import date, datetime
from argparse import ArgumentParser
from marshmallow import fields
from hashlib import sha256
from typing import Union, Tuple
from PaddockTS.filter import Filter
import json

def parse_date(s: str) -> date:
    """
    Parse an ISO date string into a `date` object.

    Args:
        s (str): A date string in “YYYY-MM-DD” format.

    Returns:
        date: The corresponding `datetime.date` object.

    Raises:
        ValueError: If the string does not match the expected format.
    """
    return datetime.strptime(s, "%Y-%m-%d").date()


def encode_date(d: date) -> str:
    """
    Encode a `date` object as an ISO date string.

    Args:
        d (date): The date to encode.

    Returns:
        str: The ISO-format date string (YYYY-MM-DD).
    """
    return d.isoformat()


def decode_date(s: str) -> date:
    """
    Decode an ISO date string into a `date` object.

    Args:
        s (str): A date string in “YYYY-MM-DD” format.

    Returns:
        date: The corresponding `datetime.date` object.
    """
    return date.fromisoformat(s)


# JSON (de)serialization config for date fields
date_config = config(
    encoder=encode_date,
    decoder=decode_date,
    mm_field=fields.Date
)

@dataclass_json
@dataclass(frozen=True)
class Query:
    """
    Represents a STAC-query specification, with automatic bounding‐box,
    datetime string, and unique stub generation for caching.

    Attributes:
    (User Defined)
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
    lat         : float     = field(metadata={'help': 'Latitude of the area of interest'})
    lon         : float     = field(metadata={'help': 'Longitude of the area of interest'})
    buffer      : float     = field(metadata={'help': 'Buffer in degrees around lat/lon'})
    start_time  : date      = field(metadata={'help': 'Start date (YYYY-MM-DD)'} | date_config)
    end_time    : date      = field(metadata={'help': 'End date (YYYY-MM-DD)'} | date_config)
    collections : list[str] = field(metadata={'help': 'Products to use for the query'})
    bands       : list[str] = field(metadata={'help': 'List of band data required'})

    filter     : Filter     = field(
                                default_factory=lambda: Filter.lt("eo:cloud_cover", 10),
                                metadata={'help': 'Expression to Refine Search'}
                            )
    crs        : str        = 'EPSG:6933'
    groupby    : str        = 'solar_day'
    resolution : int        = 10

    x         : float   = field(init=False, metadata={'help': 'Longitude of the area of interest'})
    y         : float   = field(init=False, metadata={'help': 'Latitude of the area of interest'})
    centre    : Tuple   = field(init=False, metadata={'help': 'Centre coordinate (x, y)'})
    lon_range : Tuple   = field(init=False, metadata={'help': 'Longitude range (min, max)'})
    lat_range : Tuple   = field(init=False, metadata={'help': 'Latitude range (min, max)'})
    datetime  : str     = field(init=False, metadata={'help': 'Time range string'})
    bbox      : list    = field(init=False, metadata={'help': 'Bounding box [min_lat, min_lon, max_lat, max_lon]'})

    # Post-init helpers to set derived fields
    set_x           = lambda s: object.__setattr__(s, 'x', s.lon)
    set_y           = lambda s: object.__setattr__(s, 'y', s.lat)
    set_centre      = lambda s: object.__setattr__(s, 'centre', (s.x, s.y))
    set_lat_range   = lambda s: object.__setattr__(s, 'lat_range', (s.x - s.buffer, s.x + s.buffer))
    set_lon_range   = lambda s: object.__setattr__(s, 'lon_range', (s.y - s.buffer, s.y + s.buffer))
    set_datetime    = lambda s: object.__setattr__(s, 'datetime', f'{str(s.start_time)}/{str(s.end_time)}')
    set_bbox        = lambda s: object.__setattr__(s, 'bbox', [s.lat_range[0], s.lon_range[0], s.lat_range[1], s.lon_range[1]])
    set_resolution  = lambda s: object.__setattr__(s, 'resolution', s.resolution if type(s.resolution) == tuple else (-s.resolution, s.resolution))

    def __post_init__(self: Self) -> None:
        """
        Populate all derived fields after the dataclass is initialized.
        """
        self.set_x()
        self.set_y()
        self.set_centre()
        self.set_lon_range()
        self.set_lat_range()
        self.set_datetime()
        self.set_bbox()

    def __str__(self: Self) -> str:
        """
        Serialize this Query to a pretty-printed JSON string.

        Returns:
            str: The JSON representation.
        """
        return self.to_json(indent=2)

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

        grp.add_argument("--lat",         type=float, required=True, help=flds['lat'].metadata['help'])
        grp.add_argument("--lon",         type=float, required=True, help=flds['lon'].metadata['help'])
        grp.add_argument("--buffer",      type=float, required=True, help=flds['buffer'].metadata['help'])
        grp.add_argument("--start_time",  type=parse_date, required=True, help=flds['start_time'].metadata['help'])
        grp.add_argument("--end_time",    type=parse_date, required=True, help=flds['end_time'].metadata['help'])
        grp.add_argument("--collections", nargs='+', required=True, help=flds['collections'].metadata['help'])
        grp.add_argument("--bands",       nargs='+', required=True, help=flds['bands'].metadata['help'])

        args, _ = parser.parse_known_args()

        if args.filter:
            try:
                filter_obj = Filter.from_string(args.filter)
            except Exception as e:
                raise ValueError(f"Invalid --filter value: {e}")
        else:
            filter_obj = Filter.lt("eo:cloud_cover", 10)
        return cls(
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
        "--lat", "-33.5040",
        "--lon", "148.4",
        "--buffer", "0.01",
        "--start_time", "2020-01-01",
        "--end_time", "2020-06-01",
        "--collections", "ga_s2am_ard_3", "ga_s2bm_ard_3",
        "--bands", "nbart_blue", "nbart_green", "nbart_red"
        "--filer", "'eo:cloud_cover < 10'"
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
        ]
    )


if __name__ == '__main__':

    test_query_from_cli()
    print('passed')