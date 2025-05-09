import xarray as xr
import rioxarray # dont remove this

def f(ds, fractions):
    """
    Add the fractional cover bands to the original xarray.Dataset.

    Parameters:
    ds (xarray.Dataset): The original xarray Dataset containing the satellite data.
    fractions (numpy.ndarray): The output array with fractional cover (time, bands, x, y).

    Returns:
    xarray.Dataset: The updated xarray Dataset with the new fractional cover bands.
    """
    # Create DataArray for each vegetation fraction
    bg = xr.DataArray(fractions[:, 0, :, :], coords=[ds.coords['time'], ds.coords['y'], ds.coords['x']], dims=['time', 'y', 'x'])
    pv = xr.DataArray(fractions[:, 1, :, :], coords=[ds.coords['time'], ds.coords['y'], ds.coords['x']], dims=['time', 'y', 'x'])
    npv = xr.DataArray(fractions[:, 2, :, :], coords=[ds.coords['time'], ds.coords['y'], ds.coords['x']], dims=['time', 'y', 'x'])
    
    # Assign new DataArrays to the original Dataset
    ds_updated = ds.assign(bg=bg, pv=pv, npv=npv)
    
    return ds_updated