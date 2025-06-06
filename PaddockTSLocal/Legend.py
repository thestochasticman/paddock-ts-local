from os.path import exists
from os.path import expanduser
from os import mkdir
from json import dump
from json import load

def get_config():
    path_configs = expanduser('~/.configs')
    if not exists(path_configs): mkdir(path_configs)
    path_config = f"path_configs/PaddockTSLocal"
    if not exists(path_config):
        config = {
            'out_dir': expanduser('~/Documents/PaddockTSLocal'),
            'tmp_dir': expanduser('~/Downloads/PaddockTSLocal')
        }
        dump(config, open(path_config, 'w'))
    else:
        config = load(open(path_config))
    return config


config = get_config()

from os import makedirs

OUT_DIR = config['out_dir']
TMP_DIR = config['tmp_dir']

if not exists(OUT_DIR): makedirs(OUT_DIR, exist_ok=True)
if not exists(TMP_DIR): makedirs(TMP_DIR, exist_ok=True)
