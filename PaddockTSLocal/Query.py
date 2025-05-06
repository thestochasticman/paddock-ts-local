
from dataclasses_json import dataclass_json
from dataclasses_json import config
from typing_extensions import Self
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from argparse import ArgumentParser
from datetime import datetime
from marshmallow import fields
from hashlib import sha256

def parse_date(s: str)   -> date: return datetime.strptime(s, "%Y-%m-%d").date()
def encode_date(d: date) -> str : return d.isoformat()
def decode_date(s: str)  -> date: return date.fromisoformat(s)
date_config = config(encoder=encode_date, decoder=decode_date, mm_field=fields.Date)

@dataclass_json
@dataclass(frozen=True)
class Query:
    lat         : float     = field(metadata={'help': 'Latitude of the area of interest'})
    lon         : float     = field(metadata={'help': 'Longitude of the area of interest'})
    buffer      : float     = field(metadata={'help': 'Buffer in degrees around lat/lon'})
    start_time  : date      = field(metadata={'help': 'Start date (YYYY-MM-DD)'} | date_config)
    end_time    : date      = field(metadata={'help': 'End date (YYYY-MM-DD)'} | date_config)
    collections : list[str] = field(metadata={'help': 'products to use for the query'})
    bands       : list[str] = field(metadata={'help': 'list of band data required'})

    crs        : str                     = 'utm'
    groupby    : str                     = 'solarday'
    resolution : int | tuple[int, int]   = (-10, 10)

    x         : float     = field(init=False, metadata={'help': 'Longitude of the area of interest'})
    y         : float     = field(init=False, metadata={'help': 'Latitude of the area of interest'})
    centre    : float     = field(init=False, metadata={'help': 'Centre of the Image to be retrieved from the Query'})
    lon_range : float     = field(init=False, metadata={'help': 'Range of Longitude'})
    lat_range : float     = field(init=False, metadata={'help': 'Range of Latitude'})
    datetime : float      = field(init=False, metadata={'help': 'Range of Time'})
    bbox      : float     = field(init=False, metadata={'help': 'Area of Interest'})

    set_x           = lambda s: object.__setattr__(s, 'x', s.lon)
    set_y           = lambda s: object.__setattr__(s, 'y', s.lat)
    set_centre      = lambda s: object.__setattr__(s, 'centre', (s.x, s.y))
    set_lat_range   = lambda s: object.__setattr__(s, 'lat_range', (s.x - s.buffer, s.x + s.buffer))
    set_lon_range   = lambda s: object.__setattr__(s, 'lon_range', (s.y - s.buffer, s.y + s.buffer))
    set_datetime    = lambda s: object.__setattr__(s, 'datetime', f'{str(s.start_time)}/{str(s.end_time)}')
    set_bbox        = lambda s: object.__setattr__(s, 'bbox', [s.lat_range[0], s.lon_range[0], s.lat_range[1], s.lon_range[1]])
    set_resolution  = lambda s: object.__setattr__(s, 'resolution', s.resolution if type(s.resolution) == tuple else (-s.resolution, s.resolution))

    def __str__(s: Self)->str: return s.to_json(indent=2)
    
    def get_stub(s: Self): return sha256(s.__str__().encode()).hexdigest()

    @classmethod
    def from_cli(cls):
        parser = ArgumentParser()
        query_group = parser.add_argument_group("query")

        query_group.add_argument("--lat", type=float, required=True, help=cls.__dataclass_fields__['lat'].metadata['help'])
        query_group.add_argument("--lon", type=float, required=True, help=cls.__dataclass_fields__['lon'].metadata['help'])
        query_group.add_argument("--buffer", type=float, required=True, help=cls.__dataclass_fields__['buffer'].metadata['help'])
        query_group.add_argument("--start_time", type=parse_date, required=True, help=cls.__dataclass_fields__['start_time'].metadata['help'])
        query_group.add_argument("--end_time", type=parse_date, required=True, help=cls.__dataclass_fields__['end_time'].metadata['help'])
        query_group.add_argument("--collections", nargs='+', required=True, help=cls.__dataclass_fields__['collections'].metadata['help'])
        query_group.add_argument("--bands", nargs='+', required=True, help=cls.__dataclass_fields__['bands'].metadata['help'])

        args, _ = parser.parse_known_args()
        return cls(
            lat=args.lat,
            lon=args.lon,
            buffer=args.buffer,
            start_time=args.start_time,
            end_time=args.end_time,
            collections=args.collections,
            bands=args.bands,
        )
    def __post_init__(s: Self)->None:
        s.set_x()
        s.set_y()
        s.set_centre()
        s.set_lat_range()
        s.set_lon_range()
        s.set_datetime()
        s.set_bbox()
        s.set_resolution()

def t():
    query = Query.from_cli()
    print(query)

if __name__ == '__main__': t()
