"""Download Soil and Landscape Grid of Australia (SLGA) soil property tiles.

`SLGA <https://esoil.io/TERNLandscapes/Public/Pages/SLGA/index.html>`_
provides national-coverage 90 m grids of soil properties (clay, sand,
silt, organic carbon, pH, bulk density, etc.) at standard depths. Each
property × depth combination is fetched as a Cloud-Optimised GeoTIFF
clipped to ``query.bbox`` and a quick-look PNG is rendered alongside.

Requires a TERN API key configured at ``~/.config/PaddockTS.json``
(``"tern_api_key": "..."``).
"""

from .download_cog import download_cog
from PaddockTS.query import Query
from .slgasoils import SLGASoils
from itertools import product
from itertools import starmap
from os import makedirs
from .plot import plot

slgasoils = SLGASoils()
get_filename = lambda q, v, d: f'{q.tmp_dir}/Environmental/{q.stub}_{v}_{d}.tif'

def download_slga_soils(query: Query, vars=['Clay', 'Sand', 'Silt'], depths=['5-15cm']):
    """Fetch the cross-product of SLGA ``vars × depths`` for ``query.bbox``.

    For each (variable, depth) pair, downloads the clipped GeoTIFF to
    ``{query.tmp_dir}/Environmental/{query.stub}_{var}_{depth}.tif`` and
    a matching quick-look PNG to ``{query.out_dir}``.

    Args:
        query: The :class:`PaddockTS.query.Query`.
        vars: SLGA variable names to fetch (e.g. ``'Clay'``, ``'Sand'``,
            ``'Silt'``, ``'SOC'``, ``'pH'``, ``'BDW'``). Default is
            ``['Clay', 'Sand', 'Silt']`` — the soil texture triple.
        depths: SLGA standard depth slices to fetch
            (e.g. ``'0-5cm'``, ``'5-15cm'``, ``'15-30cm'``,
            ``'30-60cm'``, ``'60-100cm'``, ``'100-200cm'``).
            Default ``['5-15cm']``.
    """
    makedirs(f'{query.tmp_dir}/Environmental', exist_ok=True)
    args = [(query.bbox, v, d, get_filename(query, v, d)) for v, d in product(vars, depths)]
    list(starmap(download_cog, args))
    list(starmap(plot, args))

def test():
    from PaddockTS.utils import get_example_query
    download_slga_soils(get_example_query())

if __name__ == '__main__':
    test()