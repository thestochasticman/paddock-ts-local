from argparse import ArgumentParser
from typing_extensions import Self
from dataclasses import dataclass
from dataclasses import MISSING
from dataclasses import fields
from dataclasses import field
from datetime import datetime
from os.path import abspath
from datetime import date

def parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, '%Y-%m-%d').date()

@dataclass(frozen=True)
class Args:
    lat         : float = field(metadata={'help': 'Latitude of the area of interest'})
    lon         : float = field(metadata={'help': 'Longitude of the area of interest'})
    buffer      : float = field(metadata={'help': 'Buffer in degrees around lat/lon'})
    start_time  : date = field(metadata={'help': 'Start date (YYYY-MM-DD)'})
    end_time    : date = field(metadata={'help': 'End date (YYYY-MM-DD)'})
    out_dir      : str = field(metadata={'help': 'Output dir for storing files'}, default='Data/shelter')
    stub        : str = field(metadata={'help': 'Stub name for the file naming'}, default='Demo')


    x           = property(lambda s: s.lon)
    y           = property(lambda s: s.lat)
    centre      = property(lambda s: (s.x, s.y))
    lat_range   = property(lambda s: (s.x - s.buffer, s.lat + s.buffer))
    lon_range   = property(lambda s: (s.lon - s.buffer, s.lon + s.buffer))
    time        = property(lambda s: (s.start_time, s.end_time))

    query = property(
        lambda s: {
            'centre': s.centre,
            'y': s.y,
            'x': s.x,
            'time': s.time,
            'resolution': (-10, 10),
            'output_crs': 'epsg:6933',
            'group_by': 'solar_day'
        }
    )

    @staticmethod
    def from_cli()->Self:
        parser = ArgumentParser(description='Parse arguments for DEA Sentinel data download')

        for field_ in fields(Args):
            field_type = field_.type
            is_property = isinstance(getattr(Args, field_.name, None), property)
            default = field_.default or field_.default_factory
            if not is_property:
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
        args
        return Args(**vars(args))

args = Args.from_cli()
print(args)
print(args.query)