from .load_tern_api import load_tern_api
import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds

# SLGA soil attribute codes and their COG base URLs
# Format: {attribute_code}_{depth_start}_{depth_end}_EV_N_P_AU_NAT_C.tif
SLGA_ATTRIBUTES = {
    "Clay": "CLY",
    "Silt": "SLT",
    "Sand": "SND",
    "pH_CaCl2": "PHC",
    "Bulk_Density": "BDW",
    "Available_Water_Capacity": "AWC",
    "Effective_Cation_Exchange_Capacity": "ECE",
    "Total_Nitrogen": "NTO",
    "Total_Phosphorus": "PTO",
    "Organic_Carbon": "SOC",
    "Depth_of_Soil": "DES",
}

# Depth codes matching GlobalSoilMap specifications
DEPTH_CODES = {
    "0-5cm": ("000", "005"),
    "5-15cm": ("005", "015"),
    "15-30cm": ("015", "030"),
    "30-60cm": ("030", "060"),
    "60-100cm": ("060", "100"),
    "100-200cm": ("100", "200"),
}


def _setup_tern_auth(api_key: str) -> None:
    """Configure GDAL environment for TERN API authentication."""
    os.environ['GDAL_HTTP_USERPWD'] = f"apikey:{api_key}"


def get_cog_url(attribute: str, depth: str) -> str:
    """
    Construct the COG URL for a given soil attribute and depth.
    
    Parameters
    ----------
    attribute : str
        Soil attribute name (e.g., "Clay", "Sand", "Bulk_Density")
    depth : str
        Depth range (e.g., "5-15cm", "30-60cm")
    
    Returns
    -------
    str
        URL to the COG file (without auth - use with /vsicurl/ prefix)
    """
    attr_code = SLGA_ATTRIBUTES.get(attribute)
    if attr_code is None:
        raise ValueError(f"Unknown attribute: {attribute}. Options: {list(SLGA_ATTRIBUTES.keys())}")
    
    depth_codes = DEPTH_CODES.get(depth)
    if depth_codes is None:
        raise ValueError(f"Unknown depth: {depth}. Options: {list(DEPTH_CODES.keys())}")
    
    depth_start, depth_end = depth_codes
    
    # v2 filename format: {ATTR}_{START}_{END}_EV_N_P_AU_TRN_N_20210902.tif
    filename = f"{attr_code}_{depth_start}_{depth_end}_EV_N_P_AU_TRN_N_20210902.tif"
    
    url = f"https://data.tern.org.au/model-derived/slga/NationalMaps/SoilAndLandscapeGrid/{attr_code}/v2/{filename}"
    
    return url


def download_cog(
    bbox: tuple,
    attribute: str,
    depth: str,
    filename: str,
    api_key: str = None,
    verbose: bool = True
) -> None:
    """
    Download a subset of SLGA data from a COG.
    
    Only downloads the tiles needed for your bounding box - much faster than WCS!
    
    Parameters
    ----------
    bbox : tuple
        Bounding box as (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
    attribute : str
        Soil attribute name (e.g., "Clay", "Sand")
    depth : str
        Depth range (e.g., "5-15cm")
    filename : str
        Output filename for the GeoTIFF
    api_key : str, optional
        TERN API key
    verbose : bool
        Print progress messages
    """
    if api_key is None:
        api_key = load_tern_api()
    
    _setup_tern_auth(api_key)
    
    url = get_cog_url(attribute, depth)
    vsicurl_url = f"/vsicurl/{url}"
    
    if verbose:
        print(f"Downloading {attribute} at {depth}...")
        print(f"URL: {url}")
    
    with rasterio.open(vsicurl_url) as src:
        # Create a window from the bounding box
        window = from_bounds(*bbox, src.transform)
        
        # Read only the data we need
        data = src.read(1, window=window)
        
        # Get the transform for the windowed data
        window_transform = src.window_transform(window)
        
        # Write to output file
        profile = src.profile.copy()
        profile.update({
            'height': data.shape[0],
            'width': data.shape[1],
            'transform': window_transform,
        })
        
        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(data, 1)
    
    if verbose:
        print(f"Saved to {filename}")


if __name__ == '__main__':
    bbox = (147.35, -35.12, 147.36, -35.11)  # ~1km x 1km

    download_cog(
        bbox=bbox,
        attribute="Clay",
        depth="5-15cm",
        filename="test_clay.tif",
        verbose=True
    )