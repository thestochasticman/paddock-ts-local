import numpy as np
import xarray as xr

def calculate_indices(ds, indices):
    """
    Calculate multiple indices and add them to the dataset as new variables.
    
    Parameters:
        ds (xarray.Dataset): The input xarray dataset with dimensions (time, y, x).
        indices (dict): A dictionary where keys are the names of the indices to be added,
                        and values are functions that calculate the index.
    
    Returns:
        xarray.Dataset: A copy of the dataset with the additional indices added as new variables.
    """
    ds_updated = ds.copy()  # Work on a copy so the original remains unchanged

    for index_name, index_func in indices.items():
        # Calculate the index from the dataset
        index_data = index_func(ds_updated)
        # Add the calculated index as a new variable in the dataset
        ds_updated[index_name] = index_data
        print(f"{index_name} has shape: {index_data.shape}")
    
    return ds_updated

def calculate_ndvi(ds):
    """
    Calculate NDVI (Normalized Difference Vegetation Index) using the red and NIR bands.
    
    NDVI = (NIR - Red) / (NIR + Red)
    """
    red = ds['nbart_red']
    nir = ds['nbart_nir_1']
    ndvi = (nir - red) / (nir + red)
    return ndvi

def calculate_cfi(ds):
    """
    Calculate the Canola Flower Index (CFI).
    
    Based on Tian et al. 2022 Remote Sensing.
    Requires that NDVI is already calculated and present as 'NDVI' in the dataset.
    """
    ndvi = ds['NDVI']  # Assumes NDVI is already available in the dataset
    red = ds['nbart_red']
    green = ds['nbart_green']
    blue = ds['nbart_blue']
    
    sum_red_green = red + green
    diff_green_blue = green - blue
    cfi = ndvi * (sum_red_green + diff_green_blue)
    return cfi

def calculate_nirv(ds):
    """
    Calculate the Near Infrared Reflectance of Vegetation (NIRv).
    
    NIRv = NDVI * NIR
    """
    ndvi = ds['NDVI']  # Again, assumes NDVI has been computed
    nir = ds['nbart_nir_1']
    nirv = ndvi * nir
    return nirv

def calculate_dnirv(ds):
    """
    Calculate the difference in NIRv compared to the previous time step.
    
    Note: This will result in one fewer time step in the output.
    """
    nirv = calculate_nirv(ds)
    dnirv = nirv.diff(dim='time', n=1)
    return dnirv

def calculate_ndti(ds):
    """
    Calculate the Normalized Difference Tillage Index (NDTI).
    
    NDTI = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
    where SWIR1 and SWIR2 correspond to 'nbart_swir_2' and 'nbart_swir_3' respectively.
    """
    swir1 = ds['nbart_swir_2']
    swir2 = ds['nbart_swir_3']
    ndti = (swir1 - swir2) / (swir1 + swir2)
    return ndti

def calculate_cai(ds):
    """
    Calculate the Cellulose Absorption Index (CAI).
    
    CAI = 0.5 * (SWIR1 + SWIR2) - NIR
    where SWIR1 is 'nbart_swir_2', SWIR2 is 'nbart_swir_3', and NIR is 'nbart_nir_1'.
    """
    swir1 = ds['nbart_swir_2']
    swir2 = ds['nbart_swir_3']
    nir = ds['nbart_nir_1']
    cai = 0.5 * (swir1 + swir2) - nir
    return cai