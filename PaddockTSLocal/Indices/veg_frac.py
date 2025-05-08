import tensorflow as tf
from fractionalcover3 import unmix_fractional_cover
from fractionalcover3 import data
import numpy as np
import xarray as xr


def calculate_fractional_cover(ds, band_names, i, correction=True):
    """
    Calculate the fractional cover using specified bands from an xarray Dataset.

    Parameters:
    ds (xarray.Dataset): The input xarray Dataset containing the satellite data.
    band_names (list): A list of 6 band names to use for the calculation.
    i (int): The integer specifying which pretrained model to use.

    Returns:
    numpy.ndarray: The output array with fractional cover (time, bands, x, y).
    """
    # Check if the number of band names is exactly 6
    if len(band_names) != 6:
        raise ValueError("Exactly 6 band names must be provided")
    
    # Extract the specified bands and stack them into a numpy array with shape (time, bands, x, y)
    inref = np.stack([ds[band].values for band in band_names], axis=1)
    print('SHape of input (should be time, bands, x, y):', inref.shape)  # This should now be (time, bands, x, y)

    if correction:
        print('Using correction factors that attempt to fudge S2 data to better match Landsat.. be careful?')
        # Array for correction factors 
        # This is taken from here: https://github.com/petescarth/fractionalcover/blob/main/notebooks/ApplyModel.ipynb
        # and described in a paper by Neil Floodfor taking Landsat to Sentinel 2 reflectance (and visa versa).
        # NOT SURE THIS IS BEING IMPLEMENTD PROPERLY> THINK ABOUT ORDER OF OPERATION CT LINKED NOTEBOOK
        correction_factors = np.array([0.9551, 1.0582, 0.9871, 1.0187, 0.9528, 0.9688]) + \
                             np.array([-0.0022, 0.0031, 0.0064, 0.012, 0.0079, -0.0042])
    
        # print('Correction factors:', correction_factors)
        # print(correction_factors[:, np.newaxis, np.newaxis])
    
        # Apply correction factors using broadcasting
        inref = inref * correction_factors[:, np.newaxis, np.newaxis]
    else:
        print('Not applying correction factors')
        inref = inref * 0.0001 # if not applying the correcion factors below

    # Initialize an array to store the fractional cover results
    fractions = np.empty((inref.shape[0], 3, inref.shape[2], inref.shape[3]))

    # Loop over each time slice and apply the unmix_fractional_cover function
    for t in range(inref.shape[0]):
        fractions[t] = unmix_fractional_cover(inref[t], fc_model=data.get_model(n=i))
    
    return fractions
# # Example usage:
# band_names = ['nbart_blue', 'nbart_green', 'nbart_red', 'nbart_nir_2', 'nbart_swir_2', 'nbart_swir_3']
# i = 1  # or whichever model index you want to use
# fractions = calculate_fractional_cover(ds, band_names, i)


def add_fractional_cover_to_ds(ds, fractions):
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

# # Example usage
# ds_updated = add_fractional_cover_to_ds(ds, fractions)
# print(ds_updated)