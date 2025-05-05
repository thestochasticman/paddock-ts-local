from argparse import ArgumentParser
from typing_extensions import Self
from dataclasses import dataclass
from os.path import expanduser
from dataclasses import field
from os.path import join
from os import makedirs

url = "https://dl.fbaipublicfiles.com/segment_anything/"

@dataclass(frozen=True)
class Config:
    checkpoint          : str = 'sam_vit_h_4b8939.pth'
    type                : str = 'vit_h'
    dir                 : str = expanduser('~/SamGeo')
    min_area_ha         : float = 10
    max_area_ha         : float = 1500
    max_perimeter_ratio : float = 30
    path                : str = field(init=False)
    url                 : str = field(init=False)
    
    def __post_init__(s: Self):
        object.__setattr__(s, 'path', join(s.dir, s.checkpoint))
        object.__setattr__(s, 'url', join(url, s.checkpoint))
        makedirs(s.dir, exist_ok=True)


    @classmethod
    def from_cli(cls):
        parser = ArgumentParser()
        parser.add_argument('--sam_geo_model_dir', type=str, default=expanduser('~/SamGeo'))
        parser.add_argument('--sam_geo_model_checkpoint', type=str, default='sam_vit_h_4b8939.pth')
        parser.add_argument('--sam_geo_model_type', type=str, default='vit_h')
        parser.add_argument('--min_area_ha', type=float, default=30)
        parser.add_argument('--max_area_ha', type=float, default=1500)
        parser.add_argument('--max_perimeter_ratio', type=float, default=30)
        args, _ = parser.parse_known_args()
        print(args.min_area_ha)
        return cls(
            checkpoint=args.sam_geo_model_checkpoint,
            type=args.sam_geo_model_type,
            dir=args.sam_geo_model_dir,
            min_area_ha=args.min_area_ha,
            max_area_ha=args.max_area_ha,
            max_perimeter_ratio=args.max_perimeter_ratio
        )
    
def t():
    config = Config.from_cli()
    print(config)

if __name__ == '__main__': t()