from PaddockTSLocal.SamGeoPaddocks.download_weights import f as download_weights
from PaddockTSLocal.SamGeoPaddocks.config import Config
from PaddockTSLocal.Logger import Logger
from os.path import exists
from samgeo import SamGeo

def f(path_image: str, path_output: str, config: Config):
    if not exists(config.path): download_weights(config)
    model = SamGeo(model_type=config.type, checkpoint=config.path)
    model.generate(

    )
def t(): 
    pass