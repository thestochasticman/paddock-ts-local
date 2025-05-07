from PaddockTSLocal.SamGeoPaddocks.download_weights import f as download_weights
from PaddockTSLocal.SamGeoPaddocks.config import Config
from os.path import exists
from samgeo import SamGeo

def f(path: str):
    if not exists(path): download_weights(path)
    return SamGeo(model_type='vit_h', checkpoint=path)

def t():
    from os.path import join
    from os import getcwd
    checkpoint = 'sam_vit_h_4b8939.pth'
    path: str=join(getcwd(), 'Data', 'SamGeo', 'Model', checkpoint)
    f(path)
if __name__ == '__main__':
    t()