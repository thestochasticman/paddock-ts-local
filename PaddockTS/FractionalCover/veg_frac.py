
import numpy as np
import xarray as xr

from tensorflow.lite.python.interpreter import Interpreter 
from fractionalcover3.unmixcover import unmix_fractional_cover
from fractionalcover3 import data

import numpy as np
from os.path import dirname
from os.path import join


def get_model(n: int):
    models_dir = join(dirname(__file__), 'resources')
    available_models = [
        join(models_dir, "fcModel_32x32x32.tflite"),
        join(models_dir, "fcModel_64x64x64.tflite"),
        join(models_dir, "fcModel_256x64x256.tflite"),
        join(models_dir, "fcModel_256x128x256.tflite")
    ]
    return Interpreter(model_path=available_models[n-1])

# def unmix_fractional_cover(surface_reflectance, fc_model: Interpreter, inNull=0, outNull=0):
#     """
#     Unmixes an array of surface reflectance.

#     :param surface_reflectance: The surface reflectance data organized in a 3D array
#         with shape (nbands, nrows, ncolumns). There should be 6 bands with values
#         scaled between 0 and 1.
#     :type surface_reflectance: numpy.ndarray
#     :param fc_model: The TensorFlow Lite model interpreter, which should be initialized
#         as shown in the included code block.
#     :type fc_model: tflite_runtime.interpreter.Interpreter
#     :param inNull: The null value for the input image. Values in the input array
#         equal to this will be replaced by `outNull`.
#     :type inNull: float, optional
#     :param outNull: The null value to replace in the output array.
#     :type outNull: float, optional

#     :return: A 3D array where the first layer corresponds to bare ground,
#         the second layer corresponds to green vegetation, and the third layer
#         corresponds to non-green vegetation.

#     """
#     # Drop the Blue band. Blue is yukky
#     inshape = surface_reflectance[1:].shape
#     # reshape and transpose so it is (nrow x ncol) x 5
#     ref_data = np.reshape(surface_reflectance[1:], (inshape[0], -1)).T

#     # Run the prediction
#     inputDetails = fc_model.get_input_details()
#     outputDetails = fc_model.get_output_details()
#     fc_model.resize_tensor_input(inputDetails[0]['index'], ref_data.shape)
#     fc_model.allocate_tensors()
#     fc_model.set_tensor(inputDetails[0]['index'], ref_data.astype(np.float32))
#     fc_model.invoke()
#     fc_layers = fc_model.get_tensor(outputDetails[0]['index']).T
#     output_fc = np.reshape(fc_layers, (3, inshape[1], inshape[2]))
#     # now do the null value swap
#     output_fc[output_fc == inNull] = outNull
#     return output_fc

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