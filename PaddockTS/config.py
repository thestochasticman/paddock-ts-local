from attrs import frozen, field, Factory as F
from os.path import expanduser
from os.path import exists
from typing import Optional
from json import load
import os

@frozen
class Config:
    out_dir: str
    tmp_dir: str
    silo_dir: str = field(default=F(lambda s: f'{s.tmp_dir}/SILO', takes_self=True))


_out = expanduser('~/Downloads/PaddockTS-Outputs')
_tmp = expanduser('~/Documents/PaddockTS-Tmp')
_default = Config(_out, _tmp)

confpath = os.path.expanduser('~/.config/PaddockTS.json')
config = Config(**load(confpath)) if exists(confpath) else _default
