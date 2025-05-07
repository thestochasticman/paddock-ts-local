from PaddockTSLocal.SamGeoPaddocks.config import Config
from os.path import dirname
from os.path import join
from os import makedirs
from os import getcwd
import wget

checkpoint = 'sam_vit_h_4b8939.pth'

def f(path: str):
    makedirs(dirname(path), exist_ok=True)
    checkpoint = 'sam_vit_h_4b8939.pth'
    url = f"https://dl.fbaipublicfiles.com/segment_anything/{checkpoint}"
    wget.download(url, out=path)

def t():
    path: str=join(getcwd(), 'Data', 'SamGeo', 'Model', checkpoint)
    f(path)

if __name__ == '__main__':
    t()
