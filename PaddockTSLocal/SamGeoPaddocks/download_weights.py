from PaddockTSLocal.SamGeoPaddocks.config import Config
import wget

def f(config: Config = Config()): wget.download(config.model_url, out=config.path_model)
