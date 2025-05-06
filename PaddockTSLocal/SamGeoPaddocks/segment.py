from PaddockTSLocal.SamGeoPaddocks.download_weights import f as download_weights
from PaddockTSLocal.SamGeoPaddocks.load_model import f as load_model
from PaddockTSLocal.SamGeoPaddocks.config import Config
from os.path import exists
from samgeo import SamGeo

def f(
        path_image: str,
        path_output: str,
        config: Config,
    ):
    model = load_model(config)
    model.generate()

def t(): 
    pass