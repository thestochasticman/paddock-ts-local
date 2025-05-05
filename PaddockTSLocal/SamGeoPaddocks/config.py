from argparse import ArgumentParser
from typing_extensions import Self
from dataclasses import dataclass
from os.path import expanduser
from dataclasses import field
from os.path import join



url = "https://dl.fbaipublicfiles.com/segment_anything/"

@dataclass(frozen=True)
class Config:
    name        :  str = 'sam_vit_h_4b8939.pth'
    dir         : str = expanduser('~/SamGeo')
    path        : str = field(init=False)
    url         : str = field(init=False)

    def __post_init__(s: Self):
        object.__setattr__(s, 'path', join(s.dir, s.name))
        object.__setattr__(s, 'url', join(url, s.name))

    @classmethod
    def from_cli(cls):
        parser = ArgumentParser()
        parser.add_argument('--sam_geo_model_dir', type=str, default=expanduser('~/SamGeo'))
        parser.add_argument('--sam_geo_model_name', type=str, default='sam_vit_h_4b8939.pth')
        args, _ = parser.parse_known_args()
        return cls(name=args.sam_geo_model_name, dir=args.sam_geo_model_dir)
    
def t():
    config = Config.from_cli()
    print(config)

if __name__ == '__main__': t()