from attrs import frozen, field, Factory as F
from typing_extensions import Self
from os.path import expanduser
from typing import Optional
from os.path import exists
from os import makedirs
from json import load
import os

@frozen
class Config:
    out_dir: str
    tmp_dir: str
    email: Optional[str] = None
    tern_api_key: Optional[str] = None

    def __post_init__(s: Self):
        makedirs(s.out_dir, exist_ok=True)
        makedirs(s.tmp_dir, exist_ok=True)

_out = expanduser('~/Documents/PaddockTS-Outputs')
_tmp = expanduser('~/Downloads/PaddockTS-Tmp')
_default = Config(_out, _tmp, None)

confpath = os.path.expanduser('~/.config/PaddockTS.json')
config = Config(**load(open(confpath))) if exists(confpath) else _default

if __name__ == '__main__':
    print(config)