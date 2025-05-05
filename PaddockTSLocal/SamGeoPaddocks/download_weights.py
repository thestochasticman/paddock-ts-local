from PaddockTSLocal.SamGeoPaddocks.config import Config
from os.path import exists
import wget

def f(config: Config = Config()): wget.download(config.url, out=config.path)

def t(): return [f(), exists(Config().path)][1]

if __name__ == '__main__': print(t())