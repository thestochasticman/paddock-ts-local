import atexit
import os
import tempfile
from os import environ
from .slgasoils import SLGASoils
from PaddockTS.config import config

slga_soils = SLGASoils()


def load_tern_api_key(api_key: str = None) -> str:
    api_key = config.tern_api_key if api_key is None else api_key
    if api_key is None:
        raise ValueError('Set tern_api_key in ~/.config/PaddockTS.json or pass api_key parameter')
    return api_key


# Cache the temp file path so we only write the key to disk once per process.
_TERN_HEADER_FILE: str | None = None


def _setup_tern_auth(api_key: str) -> None:
    """Configure GDAL to authenticate against the TERN COG datastore.

    TERN requires an ``x-api-key`` HTTP header on every COG read; GDAL
    picks it up from ``GDAL_HTTP_HEADER_FILE``. We write the header to a
    process-local temp file once, register an ``atexit`` hook to clean it
    up, then point the env var at it. The legacy ``GDAL_HTTP_USERPWD``
    (basic-auth) variable is cleared so a stale value can't interfere.
    """
    global _TERN_HEADER_FILE
    if _TERN_HEADER_FILE is None:
        fd, path = tempfile.mkstemp(prefix='tern_apikey_', suffix='.txt')
        with os.fdopen(fd, 'w') as f:
            f.write(f'x-api-key: {api_key}\n')
        atexit.register(lambda p=path: os.path.exists(p) and os.remove(p))
        _TERN_HEADER_FILE = path
    environ['GDAL_HTTP_HEADER_FILE'] = _TERN_HEADER_FILE
    environ.pop('GDAL_HTTP_USERPWD', None)


def get_cog_url(attribute: str, depth: str) -> str:
    attr_code = slga_soils.attribute_codes.get(attribute)
    if attr_code is None:
        raise ValueError(
            f"Unknown SLGA attribute '{attribute}'. "
            f"Known: {sorted(slga_soils.attribute_codes)}"
        )
    if depth not in slga_soils.depth_codes:
        raise ValueError(
            f"Unknown SLGA depth '{depth}'. "
            f"Known: {sorted(slga_soils.depth_codes)}"
        )
    depth_start, depth_end = slga_soils.depth_codes[depth]
    return slga_soils.url_template.format(attr_code=attr_code, depth_start=depth_start, depth_end=depth_end)

