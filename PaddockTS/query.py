from attrs import frozen, field, Factory as F
from .config import config
from datetime import date
from hashlib import sha256

encode = lambda x: sha256(x.encode()).hexdigest()
build_from_input = F(lambda s: encode(''.join([str(s.bbox), str(s.start), str(s.end)])), takes_self=True)

@frozen
class Query:
    bbox: list[float]
    start: date
    end: date
    stub: str = field(default=build_from_input)

    tmp_dir: str = field(init=False)
    out_dir: str = field(init=False)
    centre_lon: float = field(init=False)
    centre_lat: float = field(init=False)
    sentinel2_path: str = field(init=False)
    vegfrac_path: str = field(init=False)

    tmp_dir.default(lambda s: f'{config.tmp_dir}/{s.stub}')
    out_dir.default(lambda s: f'{config.out_dir}/{s.stub}')
    centre_lon.default(lambda s: (s.bbox[0] + s.bbox[2])/2)
    centre_lat.default(lambda s: (s.bbox[1] + s.bbox[3])/2)
    sentinel2_path.default(lambda s: f'{s.tmp_dir}/{s.stub}_sentinel2.zarr')
    vegfrac_path.default(lambda s: f'{s.tmp_dir}/{s.stub}_vegfrac.zarr')

    # __str__ = lambda s: s.stub
    def __str__(s)->str: return s.stub

    @classmethod
    def from_lat_lon(cls, lat: float, lon: float, buffer_km: float, start: date, end: date, stub: str = None):
        """Create a Query from center coordinates and buffer distance in kilometers.

        Args:
            lat: Center latitude
            lon: Center longitude
            buffer_km: Buffer distance in kilometers
            start: Start date
            end: End date
            stub: Optional custom stub identifier (auto-generated if not provided)

        Returns:
            Query instance with bbox [west, south, east, north]
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


