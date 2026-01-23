"""
Download soil variables from the Soils and Landscapes Grid of Australia (SLGA).

Catalog: https://www.asris.csiro.au/arcgis/rest/services/TERN
"""
import os
import time

import numpy as np
import rioxarray as rxr
from owslib.wcs import WebCoverageService

from PaddockTS.query import Query


SLGA_URLS = {
    "Clay": "https://www.asris.csiro.au/arcgis/services/TERN/CLY_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "Silt": "https://www.asris.csiro.au/arcgis/services/TERN/SLT_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "Sand": "https://www.asris.csiro.au/arcgis/services/TERN/SND_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "pH_CaCl2": "https://www.asris.csiro.au/arcgis/services/TERN/PHC_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "Bulk_Density": "https://www.asris.csiro.au/arcgis/services/TERN/BDW_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "Available_Water_Capacity": "https://www.asris.csiro.au/arcgis/services/TERN/AWC_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "Effective_Cation_Exchange_Capacity": "https://www.asris.csiro.au/arcgis/services/TERN/ECE_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "Total_Nitrogen": "https://www.asris.csiro.au/arcgis/services/TERN/NTO_ACLEP_AU_NAT_C/MapServer/WCSServer",
    "Total_Phosphorus": "https://www.asris.csiro.au/arcgis/services/TERN/PTO_ACLEP_AU_NAT_C/MapServer/WCSServer",
}

DEPTH_IDENTIFIERS = {
    "5-15cm": '4',
    "15-30cm": '8',
    "30-60cm": '12',
    "60-100cm": '16',
}

DEFAULT_VARIABLES = ['Clay', 'Silt', 'Sand', 'pH_CaCl2', 'Bulk_Density',
                     'Available_Water_Capacity', 'Effective_Cation_Exchange_Capacity',
                     'Total_Nitrogen', 'Total_Phosphorus']

DEFAULT_DEPTHS = ['5-15cm', '15-30cm', '30-60cm', '60-100cm']


def _download_tif(bbox, url, identifier, filename):
    wcs = WebCoverageService(url, version='1.0.0')
    crs = 'EPSG:4326'
    resolution = 1
    response = wcs.getCoverage(
        identifier=identifier,
        bbox=bbox,
        crs=crs,
        format='GeoTIFF',
        resx=resolution,
        resy=resolution
    )
    with open(filename, 'wb') as file:
        file.write(response.read())


def soil_texture(query: Query, depth="5-15cm"):
    """Convert from sand, silt and clay percent to the 12 categories in the soil texture triangle."""
    outdir = query.stub_tmp_dir
    stub = query.stub

    filename_sand = os.path.join(outdir, f"{stub}_Sand_{depth}.tif")
    filename_silt = os.path.join(outdir, f"{stub}_Silt_{depth}.tif")
    filename_clay = os.path.join(outdir, f"{stub}_Clay_{depth}.tif")

    ds_sand = rxr.open_rasterio(filename_sand)
    ds_silt = rxr.open_rasterio(filename_silt)
    ds_clay = rxr.open_rasterio(filename_clay)

    sand_array = ds_sand.isel(band=0).values
    silt_array = ds_silt.isel(band=0).values
    clay_array = ds_clay.isel(band=0).values

    total_percent = sand_array + silt_array + clay_array
    sand_percent = (sand_array / total_percent) * 100
    silt_percent = (silt_array / total_percent) * 100
    clay_percent = (clay_array / total_percent) * 100

    texture = np.empty(sand_array.shape, dtype=object)

    texture[(clay_percent < 20) & (silt_percent < 50)] = 'Sandy Loam'
    texture[(sand_percent >= 70) & (clay_percent < 15)] = 'Loamy Sand'
    texture[(sand_percent >= 85) & (clay_percent < 10)] = 'Sand'
    texture[(clay_percent < 30) & (silt_percent >= 50)] = 'Silt Loam'
    texture[(clay_percent < 15) & (silt_percent >= 80)] = 'Silt'
    texture[(clay_percent >= 27) & (clay_percent < 40) & (sand_array < 20)] = 'Silty Clay Loam'
    texture[(clay_percent >= 40) & (silt_percent >= 40)] = 'Silty Clay'
    texture[(clay_percent >= 40) & (silt_percent < 40) & (sand_array < 45)] = 'Clay'
    texture[(clay_percent >= 35) & (sand_percent >= 45)] = 'Sandy Clay'
    texture[(clay_percent >= 27) & (clay_percent < 40) & (sand_array >= 20) & (sand_array < 45)] = 'Clay Loam'
    texture[(clay_percent >= 20) & (clay_percent < 35) & (sand_array >= 45) & (silt_array < 28)] = 'Sandy Clay Loam'
    texture[(clay_percent >= 15) & (clay_percent < 27) & (silt_array >= 28) & (silt_array < 50) & (sand_array < 53)] = 'Loam'

    return texture


def slga_soils(
    query: Query,
    variables: list[str] = None,
    depths: list[str] = None,
    verbose=True
):
    """Download soil variables from CSIRO at 90m resolution for region of interest.

    Parameters
    ----------
        query: Query object with lat, lon, buffer, stub, stub_tmp_dir
        variables: List of soil variables (default: all available)
        depths: List of depths (default: all available)
        verbose: Print progress messages

    Downloads
    ---------
        A Tiff file for each variable/depth combination
    """
    from os import makedirs

    if variables is None:
        variables = DEFAULT_VARIABLES
    if depths is None:
        depths = DEFAULT_DEPTHS

    lat, lon, buffer = query.lat, query.lon, query.buffer
    outdir, stub = query.stub_tmp_dir, query.stub
    makedirs(outdir, exist_ok=True)

    if verbose:
        print("Starting slga_soils")

    buffer = max(0.00001, buffer)
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]

    for depth in depths:
        identifier = DEPTH_IDENTIFIERS[depth]
        for variable in variables:
            filename = os.path.join(outdir, f"{stub}_{variable}_{depth}.tif")
            url = SLGA_URLS[variable]

            attempt = 0
            base_delay = 5
            max_retries = 3

            while attempt < max_retries:
                try:
                    _download_tif(bbox, url, identifier, filename)
                    if verbose:
                        print(f"Downloaded {filename}")
                    break
                except Exception as e:
                    if verbose:
                        print(f"Failed to download {variable} {depth}, attempt {attempt + 1} of {max_retries}", e)
                    attempt += 1
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        if verbose:
                            print(f"Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
