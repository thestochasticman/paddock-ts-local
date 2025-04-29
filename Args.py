from argparse import ArgumentParser
from typing_extensions import Self
from dataclasses import dataclass
from dataclasses import MISSING
from dataclasses import fields
from datacube import Datacube
from dataclasses import field
from datetime import datetime
from os.path import exists
from os.path import abspath
from hashlib import sha256
from datetime import date
from os.path import join
from os import makedirs
from json import load
from json import dump


def parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, '%Y-%m-%d').date()

m = lambda x: [makedirs(x, exist_ok=True), x][1]

@dataclass(frozen=True)
class Args:
    lat         : float = field(metadata={'help': 'Latitude of the area of interest'})
    lon         : float = field(metadata={'help': 'Longitude of the area of interest'})
    buffer      : float = field(metadata={'help': 'Buffer in degrees around lat/lon'})
    start_time  : date  = field(metadata={'help': 'Start date (YYYY-MM-DD)'})
    end_time    : date  = field(metadata={'help': 'End date (YYYY-MM-DD)'})
    out_dir     : str   = field(metadata={'help': 'Output dir for storing files'}, default='Data/shelter')
    stub        : str   = field(metadata={'help': 'Stub name for the file naming'}, default='')
    path_out    : str   = field(init=False)
    app_name    : str   = 'Shelter'

    x           = property(lambda s: s.lon)
    y           = property(lambda s: s.lat)
    centre      = property(lambda s: (s.x, s.y))
    lat_range   = property(lambda s: (s.lat - s.buffer, s.lat + s.buffer))
    lon_range   = property(lambda s: (s.lon - s.buffer, s.lon + s.buffer))
    time        = property(lambda s: (s.start_time, s.end_time))
    bbox        = property(lambda s: [s.lon_range[0], s.lat_range[0], s.lon_range[1], s.lat_range[1]])

    query       = property(
                    lambda s: {
                        'centre': s.centre,
                        'y': s.y,
                        'x': s.x,
                        'time': s.time,
                        'buffer': s.buffer,
                        'resolution': (-10, 10),
                        'output_crs': 'epsg:6933',
                        'group_by': 'solar_day'
                    }
                )
    
    __str__             =  lambda s: str(s.query)
    __sha256__          = property(lambda s: sha256(s.__str__().encode()).hexdigest())
    unique_query_id     = property(lambda s: s.__sha256__)
    path_stubs_mapping  = property(lambda s: join(s.out_dir, 'stubs_mapping.json'))
    stubs_mapping       = property(lambda s: {} if not exists(s.path_stubs_mapping) else load(open(s.path_stubs_mapping)))
    dc                  = property(lambda s: Datacube(app=s.app_name))

    def __post_init__(s: Self):
        m(s.out_dir)
        if type(s.stub) != str:
            stubs_mapping: dict = s.stubs_mapping
            unique_query_id = s.unique_query_id
            existing_unique_query_ids = list(stubs_mapping.values())
            if unique_query_id in existing_unique_query_ids:
                for stub, unique_id in stubs_mapping.items():
                    if unique_id == unique_query_id:
                        break
                path_out = join(s.out_dir, stub)
            else:
                stub = str(len(stubs_mapping) + 1)
                stubs_mapping[stub] = unique_query_id
                dump(stubs_mapping, open(s.path_stubs_mapping, 'w'))
                path_out = join(s.out_dir, stub)
        else:
            stub = s.stub
            stubs_mapping = s.stubs_mapping
            unique_query_id = s.unique_query_id
            path_out = join(s.out_dir, stub)
            if stub not in stubs_mapping:
                stubs_mapping[stub] = unique_query_id
                dump(stubs_mapping, open(s.path_stubs_mapping, 'w'))
        


        object.__setattr__(s, 'path_out', path_out + '_ds2.pkl')
        object.__setattr__(s, 'stub', stub)

    @staticmethod
    def from_cli()->'Args':
        parser = ArgumentParser(description='Parse arguments for DEA Sentinel data download')

        for field_ in fields(Args):
            field_type = field_.type
            is_property = isinstance(getattr(Args, field_.name, None), property)
            default = field_.default or field_.default_factory
            if not is_property and field_.init:
                name = f'--{field_.name}'
                help_text = field_.metadata.get('help', '')
                no_default = field_.default is MISSING and field_.default_factory is MISSING
                field_type = field_.type
                if no_default:
                    _type = parse_date if field_type is date else field_type
                    parser.add_argument(name, type=_type, required=True, help=help_text)
                else:
                    _type = abspath if 'out_dir' in name else field_type
                    parser.add_argument(name, type=_type, required=False, help=help_text, default=default)
            
        args = parser.parse_args()
        return Args(**vars(args))

def t():
    args: Args = Args.from_cli()
    args.stub
    args.path_out
    return True

if __name__ == '__main__': print('passed' if t() else 'failed')