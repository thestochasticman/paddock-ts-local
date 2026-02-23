from json import load
from os import environ
from os.path import expanduser
from .slgasoils import SLGASoils
from PaddockTS.config import config

slga_soils = SLGASoils()

def load_tern_api_key(api_key: str=None)->str:
    api_key = config.tern_api_key if api_key is None else api_key
    if api_key is None:
        raise ValueError('Set tern_api_key in ~/.config/PaddockTS.json or pass api_key parameter')
    return api_key

def _setup_tern_auth(api_key: str) -> None:
    environ.update({'GDAL_HTTP_USERPWD': f'apikey:{api_key}'})

def get_cog_url(attribute: str, depth: str) -> str:
    attr_code = slga_soils.attribute_codes.get(attribute)
    depth_start, depth_end = slga_soils.depth_codes.get(depth)
    return slga_soils.url_template.format(attr_code=attr_code, depth_start=depth_start, depth_end=depth_end)

