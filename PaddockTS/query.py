from attrs import frozen, field, Factory as F
from typing_extensions import Self
from .config import config
from hashlib import sha256
from datetime import date
from os import makedirs

encode = lambda x: sha256(x.encode()).hexdigest()
build_from_input = F(lambda s: encode(''.join([str(s.bbox), str(s.start), str(s.end)])), takes_self=True)

@frozen
class Query:
    """A request to run the pipeline over a region and time range.

    Every pipeline stage takes a ``Query`` and writes outputs to paths
    derived from it (``tmp_dir``, ``out_dir``, ``sentinel2_path``,
    ``fractional_cover_path``). The object is immutable and hashable;
    re-running with the same inputs yields the same ``stub`` and reuses
    cached files on disk.

    Attributes:
        bbox: Bounding box ``[west, south, east, north]`` in EPSG:4326
            (decimal degrees).
        start: Inclusive start date.
        end: Inclusive end date.
        stub: Short identifier used in every output filename. Defaults to
            a SHA-256 hash of ``(bbox, start, end)`` so repeat runs collide
            with their cached outputs. Pass an explicit string for
            human-readable filenames.
        tmp_dir: Per-query intermediates directory
            (``{config.tmp_dir}/{stub}``). Created on init.
        out_dir: Per-query final-outputs directory
            (``{config.out_dir}/{stub}``). Created on init.
        centre_lon: Centre longitude of ``bbox`` (derived).
        centre_lat: Centre latitude of ``bbox`` (derived).
        sentinel2_path: Path to the Sentinel-2 Zarr written by
            :func:`PaddockTS.Sentinel2.download_sentinel2`.
        fractional_cover_path: Path to the fractional-cover Zarr written
            by :func:`PaddockTS.FractionalCover.compute_fractional_cover`.

    Example:
        >>> from datetime import date
        >>> from PaddockTS.query import Query
        >>> q = Query(
        ...     bbox=[148.46, -34.39, 148.50, -34.36],
        ...     start=date(2023, 1, 1),
        ...     end=date(2023, 12, 31),
        ...     stub='milgadara',
        ... )
        >>> q.sentinel2_path
        '.../milgadara/milgadara_sentinel2.zarr'
    """

    bbox: list[float]
    start: date
    end: date
    stub: str = field(default=build_from_input)

    tmp_dir: str = field(init=False)
    out_dir: str = field(init=False)
    centre_lon: float = field(init=False)
    centre_lat: float = field(init=False)
    sentinel2_path: str = field(init=False)
    fractional_cover_path: str = field(init=False)

    tmp_dir.default(lambda s: f'{config.tmp_dir}/{s.stub}')
    out_dir.default(lambda s: f'{config.out_dir}/{s.stub}')
    centre_lon.default(lambda s: (s.bbox[0] + s.bbox[2])/2)
    centre_lat.default(lambda s: (s.bbox[1] + s.bbox[3])/2)
    sentinel2_path.default(lambda s: f'{s.tmp_dir}/{s.stub}_sentinel2.zarr')
    fractional_cover_path.default(lambda s: f'{s.tmp_dir}/{s.stub}_fractional_cover.zarr')

    def __post_init__(s: Self):
        makedirs(s.tmp_dir, exist_ok=True)
        makedirs(s.out_dir, exist_ok=True)

    # __str__ = lambda s: s.stub
    def __str__(s)->str: return s.stub

    @classmethod
    def from_lat_lon(cls, lat: float, lon: float, buffer_km: float, start: date, end: date, stub: str = None):
        """Build a Query from a centre point and a square buffer in kilometres.

        Convenience constructor for users who think in "X km around a point"
        rather than bounding-box corners. The km-to-degrees conversion is
        approximate (treats the Earth locally as a sphere) which is fine
        for the buffer sizes typical of paddock-scale work (≲50 km).

        Args:
            lat: Centre latitude in decimal degrees (EPSG:4326).
            lon: Centre longitude in decimal degrees (EPSG:4326).
            buffer_km: Half-side of the square buffer, in kilometres. The
                resulting bbox spans ``2 * buffer_km`` on each side.
            start: Inclusive start date.
            end: Inclusive end date.
            stub: Optional human-readable stub identifier. If omitted, a
                SHA-256 hash of the inputs is used.

        Returns:
            Query: Instance with ``bbox = [west, south, east, north]``.

        Example:
            >>> from datetime import date
            >>> q = Query.from_lat_lon(-34.38, 148.48, 2.0,
            ...                        date(2023, 1, 1), date(2023, 12, 31),
            ...                        stub='milgadara')
        """
        # Convert km to degrees (approximate)
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
        import math
        lat_buffer = buffer_km / 111.0
        lon_buffer = buffer_km / (111.0 * math.cos(math.radians(lat)))

        bbox = [
            lon - lon_buffer,  # west (minX)
            lat - lat_buffer,  # south (minY)
            lon + lon_buffer,  # east (maxX)
            lat + lat_buffer   # north (maxY)
        ]

        if stub is not None:
            return cls(bbox=bbox, start=start, end=end, stub=stub)
        else:
            return cls(bbox=bbox, start=start, end=end)


