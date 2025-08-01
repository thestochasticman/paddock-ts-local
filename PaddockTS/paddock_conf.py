from dataclasses import dataclass
from typing_extensions import Union

@dataclass
class PaddockConf:
    min_area_ha         : float = 10
    max_area_ha         : float = 1500
    max_perim_area_ratio: float = 30
    device              : str = 'cpu'

