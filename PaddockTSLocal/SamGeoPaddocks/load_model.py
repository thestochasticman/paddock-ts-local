from PaddockTSLocal.SamGeoPaddocks.download_weights import f as download_weights
from PaddockTSLocal.SamGeoPaddocks.config import Config
from os.path import exists
from samgeo import SamGeo

def f(config: Config=Config()):
    if not exists(config.path): download_weights(config)
    return SamGeo(model_type=config.type, checkpoint=config.path)

def t():
    model = f()
    print(model)
if __name__ == '__main__':
    t()