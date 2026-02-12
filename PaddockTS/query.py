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
    silo_dir: str = field(init=False)
    sentinel2_path: str = field(init=False)

    tmp_dir.default(lambda s: f'{config.tmp_dir}/{s.stub}')
    out_dir.default(lambda s: f'{config.out_dir}/{s.stub}')
    centre_lon.default(lambda s: (s.bbox[0] + s.bbox[1])/2)
    centre_lat.default(lambda s: (s.bbox[1] + s.bbox[2])/2)
    sentinel2_path.default(lambda s: f'{s.tmp_dir}/{s.stub}_sentinel2.zarr')

    # __str__ = lambda s: s.stub
    def __str__(s)->str: return s.stub

