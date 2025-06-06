from PaddockTSLocal.utils import load_pickle
from typing_extensions import Union
import geopandas as gpd
import pandas as pd
import numpy as np
import pickle
import xarray as xr
import rioxarray

def drop_oa(ds):
    """Subset the list of variables in an xarray object to then filter it"""
    l = list(ds.keys())
    l = [item for item in l if 'oa' not in item]
    print('Keeping vars:', l)
    print('Number of variables:', len(l))
    return l

def generate_paddock_ts(
        pol: Union[str, gpd.GeoDataFrame],
        ds2: Union[str, xr.Dataset],
        
    ):
    pol = gpd.read_file(pol) if isinstance(pol, str) else pol
    ds2 = load_pickle(ds2) if isinstance(ds2, str) else ds2

    keep_vars = drop_oa(ds2)

    # Make paddock-variable-time (pvt) array:
    ts = []
    for datarow in pol.itertuples(index=True):
        ds_ = ds2[keep_vars]
        ds_clipped = ds_.rio.clip([datarow.geometry])
        pol_ts = ds_clipped.where(ds_clipped > 0).median(dim=['x', 'y'])
        array = pol_ts.to_array().transpose('variable', 'time').values.astype(np.float32)
        ts.append(array[None, :])
    pvt = np.vstack(ts)

    # # save the pvt
    # np.save(outdir + stub + '_pvt', pvt, allow_pickle=True, fix_imports=True)

    # # save the list of variable names:
    # with open(outdir + stub + '_pvt_vars.pkl', 'wb') as f:
    #     pickle.dump(keep_vars, f)

    # print('Created the file: ', outdir + stub + '_pvt_vars.pkl')
    # print('Finished!')