"""Environmental data download functions."""

from .terrain_tiles import terrain_tiles
from .slga_soils import slga_soils, soil_texture, SLGA_URLS, DEFAULT_VARIABLES as SOIL_VARIABLES, DEFAULT_DEPTHS as SOIL_DEPTHS
from .ozwald_daily import ozwald_daily, OZWALD_DAILY_VARIABLES
from .ozwald_8day import ozwald_8day, OZWALD_8DAY_VARIABLES
from .silo_daily import silo_daily, SILO_VARIABLES
from .daesim_forcing import daesim_forcing, daesim_soils

__all__ = [
    'terrain_tiles',
    'slga_soils',
    'soil_texture',
    'ozwald_daily',
    'ozwald_8day',
    'silo_daily',
    'daesim_forcing',
    'daesim_soils',
    'SLGA_URLS',
    'SOIL_VARIABLES',
    'SOIL_DEPTHS',
    'OZWALD_DAILY_VARIABLES',
    'OZWALD_8DAY_VARIABLES',
    'SILO_VARIABLES',
]
