import attrs
import json
from datetime import date
from argparse import ArgumentParser
from hashlib import sha256
from os.path import expanduser
from PaddockTS.filter import Filter
from PaddockTS.utils import parse_date
from PaddockTS.config import config


def _convert_date(value: str | date) -> date:
    """Convert string to date if needed."""
    return parse_date(value) if isinstance(value, str) else value


DEFAULT_COLLECTIONS = ['ga_s2am_ard_3', 'ga_s2bm_ard_3']

DEFAULT_BANDS = [
    'nbart_blue',
    'nbart_green',
    'nbart_red',
    'nbart_red_edge_1',
    'nbart_red_edge_2',
    'nbart_red_edge_3',
    'nbart_nir_1',
    'nbart_nir_2',
    'nbart_swir_2',
    'nbart_swir_3',
]


@attrs.frozen
class Query:
    """
    Represents a STAC-query specification, with automatic bounding‐box,
    datetime string, and unique stub generation for caching.

    Attributes:
        lat (float)                 : Latitude of the area of interest
        lon (float)                 : Longitude of the area of interest
        buffer (float)              : Buffer in degrees around (lat, lon)
        start_time (date)           : Start date of query (inclusive)
        end_time (date)             : End date of query (inclusive)
        collections (list[str])     : List of STAC collection IDs
        bands (list[str])           : List of band names to load
        crs (str)                   : Coordinate reference system (default "EPSG:6933")
        groupby (str)               : ODC groupby key (default "solar_day")
        resolution (int)            : Spatial resolution in metres (default 10)
        filter (Filter)             : Expression to refine search
        stub (str)                  : Name of the directory where results are stored
        out_dir (str)               : Output directory path
        tmp_dir (str)               : Temporary directory path

    Properties (computed):
        centre (tuple)              : (lon, lat) pair
        lon_range (tuple)           : (min_lon, max_lon)
        lat_range (tuple)           : (min_lat, max_lat)
        datetime (str)              : "YYYY-MM-DD/YYYY-MM-DD" string
        bbox (list)                 : [min_lon, min_lat, max_lon, max_lat]
        stub_tmp_dir (str)          : Temporary directory for this query
        stub_out_dir (str)          : Output directory for this query
        path_ds2 (str)              : Path to ds2.pkl
        path_preseg_tif (str)       : Path to preseg.tif
        path_polygons (str)         : Path to polygons.gpkg
        dir_checkpoint_plots (str)  : Path to checkpoints directory
    """

    lat: float
    lon: float
    buffer: float
    start_time: date = attrs.field(converter=_convert_date)
    end_time: date = attrs.field(converter=_convert_date)

    collections: list[str] = attrs.field(factory=DEFAULT_COLLECTIONS.copy)
    bands: list[str] = attrs.field(factory=DEFAULT_BANDS.copy)
    filter: Filter = attrs.field(factory=lambda: Filter.lt("eo:cloud_cover", 10))
    crs: str = 'EPSG:6933'
    groupby: str = 'solar_day'
    resolution: int = 10
    stub: str | None = None
    out_dir: str = attrs.field(factory=lambda: config.out_dir)
    tmp_dir: str = attrs.field(factory=lambda: config.tmp_dir)

    def __attrs_post_init__(self):
        if self.stub is None:
            object.__setattr__(self, 'stub', self._compute_stub())

    def _compute_stub(self) -> str:
        """Compute stub without including stub itself in the hash."""
        init_fields = {
            f.name: getattr(self, f.name)
            for f in attrs.fields(self.__class__)
            if f.name != 'stub'
        }

        def default(o):
            if isinstance(o, date):
                return o.isoformat()
            if isinstance(o, Filter):
                return o.to_dict()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        return sha256(json.dumps(init_fields, sort_keys=True, default=default).encode()).hexdigest()

    # Computed properties
    @property
    def centre(self) -> tuple[float, float]:
        return (self.lon, self.lat)

    @property
    def lon_range(self) -> tuple[float, float]:
        return (self.lon - self.buffer, self.lon + self.buffer)

    @property
    def lat_range(self) -> tuple[float, float]:
        return (self.lat - self.buffer, self.lat + self.buffer)

    @property
    def datetime(self) -> str:
        return f'{self.start_time}/{self.end_time}'

    @property
    def bbox(self) -> list[float]:
        """Bounding box in Lon/Lat: [min(lon), min(lat), max(lon), max(lat)]"""
        return [self.lon_range[0], self.lat_range[0], self.lon_range[1], self.lat_range[1]]

    @property
    def stub_tmp_dir(self) -> str:
        stub = self.stub or self._compute_stub()
        return f"{self.tmp_dir}/{stub}"

    @property
    def stub_out_dir(self) -> str:
        stub = self.stub or self._compute_stub()
        return f"{self.out_dir}/{stub}"

    @property
    def path_ds2(self) -> str:
        return f"{self.stub_tmp_dir}/ds2.pkl"

    @property
    def path_preseg_tif(self) -> str:
        return f"{self.stub_tmp_dir}/preseg.tif"

    @property
    def path_polygons(self) -> str:
        return f"{self.stub_tmp_dir}/polygons.gpkg"

    @property
    def dir_checkpoint_plots(self) -> str:
        return f"{self.stub_out_dir}/checkpoints"

    def __str__(self) -> str:
        """Serialize this Query to a pretty-printed JSON string."""
        def default(o):
            if isinstance(o, date):
                return o.isoformat()
            if isinstance(o, Filter):
                return o.to_dict()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        init_fields = {
            f.name: getattr(self, f.name)
            for f in attrs.fields(self.__class__)
        }
        return json.dumps(init_fields, indent=2, default=default)

    def get_stub(self) -> str:
        """Compute a SHA-256 hash of this Query's JSON to use as a cache key."""
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

        grp.add_argument("--stub", type=str, required=True)
        grp.add_argument("--lat", type=float, required=True)
        grp.add_argument("--lon", type=float, required=True)
        grp.add_argument("--buffer", type=float, required=True)
        grp.add_argument("--start_time", type=parse_date, required=True)
        grp.add_argument("--end_time", type=parse_date, required=True)
        grp.add_argument("--collections", nargs='+', required=True)
        grp.add_argument("--bands", nargs='+', required=True)
        grp.add_argument("--filter", type=str, required=True)

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


def get_example_query() -> Query:
    """Return a sample Query for testing or demonstration."""
    return Query(
        stub='test_example_query',
        lat=-33.5040,
        lon=148.4,
        buffer=0.01,
        start_time=date(2020, 1, 1),
        end_time=date(2020, 6, 1),
    )
