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
from os import makedirs
from os.path import exists
from .plot import plot

slgasoils = SLGASoils()
get_filename = lambda q, v, d: f'{q.tmp_dir}/Environmental/{q.stub}_{v}_{d}.tif'


def _cog_is_cached(tif: str) -> bool:
    """A SLGA COG is considered cached when both the .tif and its sibling
    ._SUCCESS marker exist. The marker is written *after* the download
    completes (see below) so a kill-9 mid-write leaves the cache invalid
    and the next call re-fetches cleanly — same contract used elsewhere
    in PaddockTS for terrain, sentinel2, etc."""
    return exists(tif) and exists(f'{tif}._SUCCESS')


def download_slga_soils(query: Query, vars=['Clay', 'Sand', 'Silt'], depths=['5-15cm']):
    """Fetch the cross-product of SLGA ``vars × depths`` for ``query.bbox``.

    For each (variable, depth) pair, downloads the clipped GeoTIFF to
    ``{query.tmp_dir}/Environmental/{query.stub}_{var}_{depth}.tif`` and
    a matching quick-look PNG sibling. Cached per (var, depth): if the
    TIF + ``._SUCCESS`` marker already exist, the download is skipped;
    if the PNG already exists, the plot is skipped.

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
    for v, d in product(vars, depths):
        tif = get_filename(query, v, d)
        png = tif[:-4] + '.png'   # plot() writes sibling .png

        if _cog_is_cached(tif):
            print(f'  SLGA {v} {d}: cached at {tif}')
        else:
            download_cog(query.bbox, v, d, tif)
            # Touch _SUCCESS *after* the COG write completes so the cache
            # is only considered valid when the download finished cleanly.
            with open(f'{tif}._SUCCESS', 'w') as f:
                f.write('')

        if not exists(png):
            plot(query.bbox, v, d, tif)

def test():
    from PaddockTS.utils import get_example_query
    download_slga_soils(get_example_query())

if __name__ == '__main__':
    test()