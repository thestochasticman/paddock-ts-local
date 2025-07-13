from os.path import expanduser
from os.path import exists
from json import dump
from json import load
from os import mkdir

def get_config():
    path_configs = expanduser('~/.configs')
    if not exists(path_configs): mkdir(path_configs)
    path_config = f"{path_configs}/PaddockTSLocal.json"
    if not exists(path_config):
        config = {
            'out_dir': expanduser('~/Documents/PaddockTSLocal'),
            'tmp_dir': expanduser('~/Downloads/PaddockTSLocal'),
            'scratch_dir': expanduser('~/Scratch/PaddockTSLocal'),
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
NDWI_FOURIER_GEOTIFF_DIR = f"{TMP_DIR}/NDWI_FOURIER_GEOTIFF"

SAMGEO_DIR = f"{TMP_DIR}/SAMGEO"
SAMGEO_OUTPUT_MASK_DIR = f"{SAMGEO_DIR}/OUTPUT_MASK"
SAMGEO_OUTPUT_VECTOR_DIR = f"{SAMGEO_DIR}/OUTPUT_VECTOR"
SAMGEO_FILTERED_OUTPUT_VECTOR_DIR = f"{SAMGEO_DIR}/FILTERED_OUTPUT_VECTOR"
SAMGEO_MODELS_DIR = f"{SAMGEO_DIR}/MODELS"
SAMGEO_MODEL_PATH = f"{SAMGEO_MODELS_DIR}/sam_vit_h_4b8939.pth"

DS2I_DIR = f"{TMP_DIR}/DS2I"
PADDOCK_TS_DIR = f"{TMP_DIR}/PADDOCK_TS"

SILO_DIR = f"{TMP_DIR}/SILO"

if not exists(DS2_DIR): mkdir(DS2_DIR)
if not exists(NDWI_FOURIER_GEOTIFF_DIR): mkdir(NDWI_FOURIER_GEOTIFF_DIR)
if not exists(SAMGEO_DIR): mkdir(SAMGEO_DIR)
if not exists(SAMGEO_MODELS_DIR): mkdir(SAMGEO_MODELS_DIR)
if not exists(SAMGEO_OUTPUT_MASK_DIR): mkdir(SAMGEO_OUTPUT_MASK_DIR)
if not exists(SAMGEO_OUTPUT_VECTOR_DIR): mkdir(SAMGEO_OUTPUT_VECTOR_DIR)
if not exists(SAMGEO_FILTERED_OUTPUT_VECTOR_DIR): mkdir(SAMGEO_FILTERED_OUTPUT_VECTOR_DIR)
if not exists(DS2I_DIR): mkdir(DS2I_DIR)
if not exists(PADDOCK_TS_DIR): mkdir(PADDOCK_TS_DIR)
if not exists(SILO_DIR): mkdir(SILO_DIR)