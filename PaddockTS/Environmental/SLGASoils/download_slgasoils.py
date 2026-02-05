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
    makedirs(f'{query.tmp_dir}/Environmental', exist_ok=True)
    args = [(query.bbox, v, d, get_filename(query, v, d)) for v, d in product(vars, depths)]
    list(starmap(download_cog, args))
    list(starmap(plot, args))

def test():
    from PaddockTS.utils import get_example_query
    download_slga_soils(get_example_query())

if __name__ == '__main__':
    test()