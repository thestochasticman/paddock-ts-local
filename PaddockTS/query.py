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

    tmp_dir: str = config.tmp_dir
    out_dir: str = config.out_dir
    centre_lon: float = field(init=False)
    centre_lat: float = field(init=False)
    stub_tmp_dir: str = field(init=False)
    stub_out_dir: str = field(init=False)
    silo_dir: str = field(init=False)
    
    centre_lon.default(lambda s: (s.bbox[0] + s.bbox[1])/2)
    centre_lat.default(lambda s: (s.bbox[1] + s.bbox[2])/2)
    stub_tmp_dir.default(lambda s: f'{s.tmp_dir}/{s.stub}')
    stub_out_dir.default(lambda s: f'{s.out_dir}/{s.stub}')

if __name__ == '__main__':
    query = Query([1, 2, 3, 4], date(2020, 1, 1), date(2020, 2, 2))
    print(query.centre_lon)
    print(query)