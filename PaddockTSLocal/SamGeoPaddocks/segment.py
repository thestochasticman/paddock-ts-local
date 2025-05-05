from PaddockTSLocal.SamGeoPaddocks.download_weights import f as download_weights
from PaddockTSLocal.SamGeoPaddocks.config import Config
from PaddockTSLocal.Logger import Logger
from os.path import exists

def f(config: Config, path_image: str):
    if not exists(config.path): download_weights(config)
    