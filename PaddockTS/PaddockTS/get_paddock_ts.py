from PaddockTS.legend import SAMGEO_FILTERED_OUTPUT_VECTOR_DIR
from PaddockTS.legend import PADDOCK_TS_DIR
import pickle
from PaddockTS.legend import DS2_DIR
import geopandas as gpd
import numpy as np
import pickle

def drop_oa(ds):
    """Subset the list of variables in an xarray object to then filter it"""
    l = list(ds.keys())
    l = [item for item in l if 'oa' not in item]
    # print('Keeping vars:', l)
    # print('Number of variables:', len(l))
    return l

def get_paddock_ts(stub: str):
    """
    Extract time-series for each paddock polygon and save results.
    Steps:
        1. Read filtered paddock polygons (GeoPackage) for the given stub.
        2. Assign a categorical paddock ID to each polygon.
        3. Load the corresponding DS2 dataset from pickle.
        4. Remove 'oa' variables via drop_oa().
        5. For each paddock polygon:
           a. Clip the dataset to the polygon geometry.
           b. Mask non-positive values.
           c. Compute median over spatial dims 'x' and 'y', yielding time series per variable.
           d. Stack into an array of shape (1, n_variables, n_times).
        6. Vertically concatenate all paddock arrays to form shape (n_paddocks, n_variables, n_times).
        7. Save the time-series array (.npy) and the list of variable names (.pkl).
    """
    pol = gpd.read_file(f"{SAMGEO_FILTERED_OUTPUT_VECTOR_DIR}/{stub}.gpkg")
    pol['paddock'] = range(1, len(pol) + 1)
    pol['paddock'] = pol.paddock.astype('category')

    ds2 = pickle.load(open(f"{DS2_DIR}/{stub}.pkl", 'rb'))
    keep_vars = drop_oa(ds2)

    ts = []
    for datarow in pol.itertuples(index=True):
        ds_ = ds2[keep_vars]
        ds_clipped = ds_.rio.clip([datarow.geometry])
        pol_ts = ds_clipped.where(ds_clipped > 0).median(dim=['x', 'y'])
        array = pol_ts.to_array().transpose('variable', 'time').values.astype(np.float32)
        ts.append(array[None, :])
    pvt = np.vstack(ts)

    np.save(f"{PADDOCK_TS_DIR}/{stub}_pvt", pvt, allow_pickle=True, fix_imports=True)

    with open(f"{PADDOCK_TS_DIR}/{stub}_pvt_vars.pkl", 'wb') as f:
        pickle.dump(keep_vars, f)


def test():
    from PaddockTS.query import get_example_query
    from os.path import exists
    query = get_example_query()
    get_paddock_ts(query.get_stub())
    stub = query.get_stub()
    return (
        exists(f"{PADDOCK_TS_DIR}/{stub}_pvt_vars.pkl") and 
        exists(f"{PADDOCK_TS_DIR}/{stub}_pvt.npy")
    )
if __name__ == '__main__':
    print(test())