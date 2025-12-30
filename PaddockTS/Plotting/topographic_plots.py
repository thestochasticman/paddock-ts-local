

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
from PaddockTS.query import Query

# Setting up logging
logging.basicConfig(level=logging.INFO)

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors



import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

def _style_axis_for_dark_bg(ax):
    ax.set_facecolor("none")
    ax.tick_params(axis="both", colors="white", labelcolor="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_color("white")

def _style_colorbar_for_dark_bg(cbar):
    cbar.ax.set_facecolor("none")
    cbar.ax.tick_params(colors="white", labelcolor="white")
    if cbar.outline is not None:
        cbar.outline.set_edgecolor("white")
        cbar.outline.set_linewidth(0.8)
    cbar.ax.yaxis.label.set_color("white")

def _save_axis_tight(
    ax,
    path: str,
    *,
    dpi: int = 300,
    transparent: bool = True,
    pad_inches: float = 0.01,
    clamp_px: dict | None = None,  # {"x0":..., "x1":..., "y0":..., "y1":...} in DISPLAY px
):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    bb = ax.get_tightbbox(renderer)  # display coords (px)

    if clamp_px:
        x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
        if "x0" in clamp_px and clamp_px["x0"] is not None:
            x0 = max(x0, clamp_px["x0"])
        if "x1" in clamp_px and clamp_px["x1"] is not None:
            x1 = min(x1, clamp_px["x1"])
        if "y0" in clamp_px and clamp_px["y0"] is not None:
            y0 = max(y0, clamp_px["y0"])
        if "y1" in clamp_px and clamp_px["y1"] is not None:
            y1 = min(y1, clamp_px["y1"])
        bb = Bbox.from_extents(x0, y0, x1, y1)

    bb_in = bb.transformed(fig.dpi_scale_trans.inverted())  # inches
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches=bb_in,
        pad_inches=pad_inches,
        transparent=transparent,
    )

def save_map_and_colorbar(
    *,
    out_stub: str,
    title: str,
    arr,
    extent,
    cmap,
    norm=None,
    origin="upper",
    interpolation="bilinear",
    cbar_label=None,
    cbar_ticks=None,
    cbar_ticklabels=None,
    overlay_fn=None,
    dpi=300,
):
    fig = plt.figure(figsize=(6.0, 6.0))
    fig.patch.set_alpha(0.0)

    # Keep cbar axis narrow; frontend can enforce min pixel width if needed
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.10], wspace=0.0)

    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    im = ax.imshow(
        arr,
        cmap=cmap,
        norm=norm,
        origin=origin,
        extent=extent,
        interpolation=interpolation,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if overlay_fn is not None:
        overlay_fn(ax)

    # # Create the colorbar ONCE
    # cbar = fig.colorbar(im, cax=cax)
    # if cbar_ticks is not None:
    #     cbar.set_ticks(cbar_ticks)
    # if cbar_ticklabels is not None:
    #     cbar.set_ticklabels(cbar_ticklabels)
    # if cbar_label:
    #     cbar.set_label(cbar_label)

    # Create the colorbar ONCE (pass ticks at construction)
    cbar = fig.colorbar(im, cax=cax, ticks=cbar_ticks if cbar_ticks is not None else None)

    # Force tick labels on the actual colorbar axis (vertical => y-axis)
    if cbar_ticklabels is not None:
        cbar.ax.set_yticklabels(list(cbar_ticklabels))

    if cbar_label:
        cbar.set_label(cbar_label)

    # Style after everything exists
    _style_axis_for_dark_bg(ax)
    _style_axis_for_dark_bg(cax)
    _style_colorbar_for_dark_bg(cbar)
    if cbar_label:
        cbar.set_label(cbar_label, color="white")

    # Draw once, then clamp bboxes so the axes cannot bleed into each other
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    ax_win = ax.get_window_extent(renderer)     # axes area (no labels)
    cax_win = cax.get_window_extent(renderer)

    map_path = f"{out_stub}.png"
    cbar_path = f"{out_stub}_cbar.png"

    # Map: clamp right edge to start of colorbar axes area
    _save_axis_tight(
        ax,
        map_path,
        dpi=dpi,
        transparent=True,
        pad_inches=0.01,
        clamp_px={"x1": cax_win.x0 - 1},
    )

    # Cbar: clamp left edge to end of map axes area
    _save_axis_tight(
        cax,
        cbar_path,
        dpi=dpi,
        transparent=True,
        pad_inches=0.01,
        clamp_px={"x0": ax_win.x1 + 1},
    )

    plt.close(fig)
    return map_path, cbar_path




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

def plot_topography(query: Query):
    outdir = query.stub_out_dir + '/environmental'
    tmpdir = query.stub_tmp_dir + '/environmental'


    # Load the imagery stack
    filename = os.path.join(query.path_ds2)
    with open(filename, 'rb') as file:
        ds_original = pickle.load(file)
    ds = ds_original

    # Load the paddocks
    pol = gpd.read_file(query.path_polygons)
    pol['paddock'] = range(1,len(pol)+1)
    pol['paddock'] = pol.paddock.astype('category')


    filename = os.path.join(outdir, f"{query.stub}_terrain.tif")
    with rasterio.open(filename) as src:
        dem = src.read(1)  
        transform = src.transform  
        crs = src.crs
        nodata = src.nodata 
        width = src.width 
        height = src.height 

    sigma = 10
    dem_smooth = gaussian_filter(dem.astype(float), sigma=sigma)
    filename = os.path.join(tmpdir, f"{query.stub}_terrain_smoothed.tif")
    with rasterio.open(filename, 'w', driver='GTiff', height=height, width=width,
                    count=1, dtype=dem_smooth.dtype, crs=crs, transform=transform,
                    nodata=nodata) as dst:
        dst.write(dem_smooth, 1) 
    print(f"Smoothed DEM saved to {filename}")

    # Load the terrain and calculate topographic variables
    filename = os.path.join(tmpdir, f"{query.stub}_terrain_smoothed.tif")
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
    filepath = os.path.join(tmpdir, query.stub + "_elevation_QGIS.tif")
    ds['terrain'].rio.to_raster(filepath)
    print(filepath)

    filepath = os.path.join(tmpdir, query.stub + "_topographic_index_QGIS.tif")
    ds['topographic_index'].rio.to_raster(filepath)
    print(filepath)

    # Need to specify the datatype for the aspect to save correctly
    filepath = os.path.join(tmpdir, query.stub + "_aspect_QGIS.tif")
    ds['aspect'].rio.to_raster(
        filepath,
        dtype="int64", 
        nodata=-1, 
    )
  
    print(filepath)

    filepath = os.path.join(tmpdir, query.stub + "_slope_QGIS.tif")
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
    filepath = os.path.join(outdir, query.stub + "_topography.png")

    plt.savefig(filepath, dpi=300)
    print(filepath)

        # extent already computed
    # pol_4326 already computed

    def overlay_red_polys_with_labels(text_color="black", edge="red"):
        def _fn(ax):
            pol_4326.plot(ax=ax, facecolor="none", edgecolor=edge, linewidth=1)
            for x, y, label in zip(
                pol_4326.geometry.centroid.x,
                pol_4326.geometry.centroid.y,
                pol_4326["paddock"],
            ):
                ax.text(x, y, label, fontsize=10, ha="center", va="center", color=text_color)
        return _fn

    # Elevation
    save_map_and_colorbar(
        out_stub=os.path.join(outdir, f"{query.stub}_elevation"),
        title="Elevation",
        arr=dem,
        extent=extent,
        cmap="terrain",
        norm=None,
        cbar_label="height above sea level (m)",
        overlay_fn=overlay_red_polys_with_labels(text_color="black", edge="red"),
    )

    # Accumulation (log)
    save_map_and_colorbar(
        out_stub=os.path.join(outdir, f"{query.stub}_accumulation"),
        title="Accumulation",
        arr=acc,
        extent=extent,
        cmap="cubehelix",
        norm=colors.LogNorm(1, np.nanmax(acc)),
        cbar_label="upstream cells",
        overlay_fn=overlay_red_polys_with_labels(text_color="yellow", edge="red"),
    )

    # Aspect (discrete ticks)
    arcgis_dirs = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    sequential_dirs = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    fdir_equal_spacing = np.zeros_like(fdir)
    for arcgis_dir, sequential_dir in zip(arcgis_dirs, sequential_dirs):
        fdir_equal_spacing[fdir == arcgis_dir] = sequential_dir

    save_map_and_colorbar(
        out_stub=os.path.join(outdir, f"{query.stub}_aspect"),
        title="Aspect",
        arr=fdir_equal_spacing,
        extent=extent,
        cmap="hsv",
        norm=None,
        cbar_ticks=sequential_dirs,
        cbar_ticklabels=["E", "SE", "S", "SW", "W", "NW", "N", "NE"],
        overlay_fn=overlay_red_polys_with_labels(text_color="black", edge="black"),
    )

    # Slope
    save_map_and_colorbar(
        out_stub=os.path.join(outdir, f"{query.stub}_slope"),
        title="Slope",
        arr=slope,
        extent=extent,
        cmap="Purples",
        norm=None,
        cbar_label="degrees",
        overlay_fn=overlay_red_polys_with_labels(text_color="black", edge="red"),
    )


    # -

    
def test():
    from PaddockTS.query import get_example_query

    query = get_example_query()

    plot_topography(query)

if __name__ == '__main__':
    test()