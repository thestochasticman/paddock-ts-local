from os.path import exists
from os.path import expanduser
from os import mkdir
from json import dump
from json import load

def get_config():
    path_configs = expanduser('~/.configs')
    if not exists(path_configs): mkdir(path_configs)
    path_config = f"{path_configs}/PaddockTSLocal.json"
    if not exists(path_config):
        config = {
            'out_dir': expanduser('~/Documents/PaddockTSLocal'),
            'tmp_dir': expanduser('~/Downloads/PaddockTSLocal'),
            'scratch_dir': expanduser('~/Scratch/PaddockTSLocal')
        }
        dump(config, open(path_config, 'w'))
    else:
        config = load(open(path_config))
    return config


config = get_config()
from os import makedirs

######## DIRS INITIALISATION #################################################
OUT_DIR = config['out_dir'] # DIR FOR VISUALISATION FILES
TMP_DIR = config['tmp_dir'] # DIR FOR NOT VISUALISATION FILES
SCRATCH_DIR = config['scratch_dir'] # DIR FOR BYPRODUCT FILES TO BE DELETED LATER

if not exists(OUT_DIR): makedirs(OUT_DIR, exist_ok=True)
if not exists(TMP_DIR): makedirs(TMP_DIR, exist_ok=True)

########## TMP DIRS INITIALISATION ###########################################
DS2_DIR = f"{TMP_DIR}/DS2"
NDWI_FOURIER_GEOTIFF_DIR = f"{TMP_DIR}/NDWI_FOURIER_GEOTIFF_DIR"

if not exists(DS2_DIR): mkdir(DS2_DIR)
if not exists(NDWI_FOURIER_GEOTIFF_DIR): mkdir(NDWI_FOURIER_GEOTIFF_DIR)
