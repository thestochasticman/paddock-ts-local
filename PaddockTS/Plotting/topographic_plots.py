

import os

from DAESIM_preprocess.topography import pysheds_accumulation, calculate_slope, add_numpy_band

import numpy as np
import pickle
import rioxarray as rxr
import geopandas as gpd
from scipy.ndimage import gaussian_filter
from pyproj import Transformer

import rasterio
from rasterio.enums import Resampling
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import argparse
import logging
import warnings

from PaddockTS.legend import *

# Setting up logging
logging.basicConfig(level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(description="""Generate a series of plots to check out the topography.
Example usage:
python3 Code/topographic_plots.py TEST6 /g/data/xe2/John/Data/PadSeg/ """)
    parser.add_argument('stub', type=str, help='Stub name for file naming.')
    parser.add_argument('outdir', type=str, help='Directory for final plots')
    parser.add_argument('tmpdir', type=str, help='Directory for intermediate files')
    return parser.parse_args()

def add_tiff_band(ds, variable, resampling_method, outdir, stub):
    """Add a new band to the xarray from a tiff file using the given resampling method"""
    filename = os.path.join(outdir, f"{stub}_{variable}.tif")
    array = rxr.open_rasterio(filename)
    reprojected = array.rio.reproject_match(ds, resampling=resampling_method)
    ds[variable] = reprojected.isel(band=0).drop_vars('band')
    return ds

def plot_topography(stub: str):
    outdir = OUT_DIR
    tmpdir = TMP_DIR

    # Load the imagery stack
    filename = os.path.join(outdir, f"{stub}_ds2.pkl")
    with open(filename, 'rb') as file:
        ds_original = pickle.load(file)
    ds = ds_original

    # Load the paddocks
    pol = gpd.read_file(outdir+stub+'_filt.gpkg')
    pol['paddock'] = range(1,len(pol)+1)
    pol['paddock'] = pol.paddock.astype('category')

    # Load the terrain and calculate topographic variables
    filename = os.path.join(tmpdir, f"{stub}_terrain_smoothed.tif")
    grid, dem, fdir, acc = pysheds_accumulation(filename)
    slope = calculate_slope(filename)

    # Make the flow directions sequential for easier plotting later
    arcgis_dirs = np.array([1, 2, 4, 8, 16, 32, 64, 128]) 
    sequential_dirs = np.array([1, 2, 3, 4, 5, 6, 7, 8])  
    fdir_equal_spacing = np.zeros_like(fdir)  
    for arcgis_dir, sequential_dir in zip(arcgis_dirs, sequential_dirs):
        fdir_equal_spacing[fdir == arcgis_dir] = sequential_dir 

    # Align & resample & reproject the topographic variables to match the imagery stack
    ds = add_numpy_band(ds, "terrain", dem, grid.affine, Resampling.average)
    ds = add_numpy_band(ds, "topographic_index", acc, grid.affine, Resampling.max)
    ds = add_numpy_band(ds, "aspect", fdir, grid.affine, Resampling.nearest)
    ds = add_numpy_band(ds, "slope", slope, grid.affine, Resampling.average)

    # Clip everything by 1 cell because these algorithms can mess up at the boundary
    ds = ds.isel(
        y=slice(1, -1),
        x=slice(1, -1) 
    )

    # Save the layers as tiff files for viewing in QGIS
    filepath = os.path.join(tmpdir, stub + "_elevation_QGIS.tif")
    ds['terrain'].rio.to_raster(filepath)
    print(filepath)

    filepath = os.path.join(tmpdir, stub + "_topographic_index_QGIS.tif")
    ds['topographic_index'].rio.to_raster(filepath)
    print(filepath)

    # Need to specify the datatype for the aspect to save correctly
    filepath = os.path.join(tmpdir, stub + "_aspect_QGIS.tif")
    ds['aspect'].rio.to_raster(
        filepath,
        dtype="int8", 
        nodata=-1, 
    )
    print(filepath)

    filepath = os.path.join(tmpdir, stub + "_slope_QGIS.tif")
    ds['slope'].rio.to_raster(filepath)
    print(filepath)

    # Reproject to EPSG:4326 so the latitude and longitudes are more useful in the plots
    pol_4326 = pol.to_crs("EPSG:4326")
    ds_4326 = ds.rio.reproject("EPSG:4326")
    left, bottom, right, top = ds_4326.rio.bounds()
    extent = (left, right, bottom, top)

    # Start plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    (ax1, ax2), (ax3, ax4) = axes

    # Ignore warnings about the centroids not being perfect because these are just used for labelling
    warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")

    # ===== Elevation Plot =====
    im = ax1.imshow(dem, cmap='terrain', interpolation='bilinear', extent=extent)
    ax1.set_title("Elevation")
    plt.colorbar(im, ax=ax1, label='height above sea level (m)')

    # Contours
    interval = 10
    contour_levels = np.arange(np.floor(np.nanmin(dem)), np.ceil(np.nanmax(dem)), interval)
    contours = ax1.contour(dem, levels=contour_levels, colors='black',
                           linewidths=0.5, alpha=0.5, extent=extent, origin='upper')
    ax1.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')

    # Scale bar
    scalebar = AnchoredSizeBar(ax1.transData, 0.01, '1km', loc='lower left', pad=0.1,
                               color='white', frameon=False, size_vertical=0.001,
                               fontproperties=FontProperties(size=12))
    ax1.add_artist(scalebar)

    # Polygons
    pol_4326.plot(ax=ax1, facecolor='none', edgecolor='red', linewidth=1)
    for x, y, label in zip(pol_4326.geometry.centroid.x, pol_4326.geometry.centroid.y, pol_4326['paddock']):
        ax1.text(x, y, label, fontsize=10, ha='center', va='center', color='black')

    # ===== Topographic Index Plot =====
    im = ax2.imshow(acc, cmap='cubehelix', norm=colors.LogNorm(1, np.nanmax(acc)),
                    interpolation='bilinear', extent=extent)
    ax2.set_title("Accumulation")
    plt.colorbar(im, ax=ax2, label='upstream cells')

    pol_4326.plot(ax=ax2, facecolor='none', edgecolor='red', linewidth=1)
    for x, y, label in zip(pol_4326.geometry.centroid.x, pol_4326.geometry.centroid.y, pol_4326['paddock']):
        ax2.text(x, y, label, fontsize=10, ha='center', va='center', color='yellow')

    # ===== Aspect Plot =====
    im = ax3.imshow(fdir_equal_spacing, cmap="hsv", origin="upper", extent=extent)
    ax3.set_title("Aspect")
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_ticks(sequential_dirs)
    cbar.set_ticklabels(["E", "SE", "S", "SW", "W", 'NW', "N", "NE"])

    pol_4326.plot(ax=ax3, facecolor='none', edgecolor='black', linewidth=1)
    for x, y, label in zip(pol_4326.geometry.centroid.x, pol_4326.geometry.centroid.y, pol_4326['paddock']):
        ax3.text(x, y, label, fontsize=12, ha='center', va='center', color='black')

    # ===== Slope Plot =====
    im = ax4.imshow(slope, cmap="Purples", origin="upper", extent=extent)
    ax4.set_title("Slope")
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label("degrees")

    pol_4326.plot(ax=ax4, facecolor='none', edgecolor='red', linewidth=1)
    for x, y, label in zip(pol_4326.geometry.centroid.x, pol_4326.geometry.centroid.y, pol_4326['paddock']):
        ax4.text(x, y, label, fontsize=12, ha='center', va='center', color='black')

    # Add lat/lon labels
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # Save combined figure
    plt.tight_layout()
    filepath = os.path.join(outdir, stub + "_topography.png")
    plt.savefig(filepath, dpi=300)
    print(filepath)

    # -

    
