import logging
import pickle
import xarray as xr
import geopandas as gpd
from dea_tools.plotting import rgb, xr_animation
import plotting_functions as pf
import os
from PaddockTSLocal.Legend import *
import pickle
from shapely.geometry import Polygon
import cv2
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter




def custom_animator(
    ds: xr.Dataset,
    pol: Polygon,
    bands: list[str],
    output_path: str,
    width_pixels: int = 600,
    fps: int = 10,
    dpi: int = 100
):
    """
    Animate an xarray Dataset along its 'time' dimension by:
      1. Computing the per-band median composite over time,
      2. Stacking those medians into an RGB reference image,
      3. Using the global max of that reference to normalize all frames,
      4. Clipping to [0…1], converting to uint8 [0…255],
      5. Rendering with square pixels (aspect='equal').
    """
    # 1. Compute figure size & aspect ratio
    ny, nx = ds.sizes['y'], ds.sizes['x']
    aspect = ny / nx
    fig_w, fig_h = width_pixels / dpi, (width_pixels * aspect) / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.axis('off')

    left   = float(ds.x.min().values)
    right  = float(ds.x.max().values)
    bottom = float(ds.y.min().values)
    top    = float(ds.y.max().values)
    
    # 2. Compute the median composite reference and its max
    #    to serve as our fixed normalizer.
    red_med   = ds['nbart_red'  ].median(dim='time').values.astype(float)
    green_med = ds['nbart_green'].median(dim='time').values.astype(float)
    blue_med  = ds['nbart_blue' ].median(dim='time').values.astype(float)

    ref_rgb = np.dstack((red_med, green_med, blue_med))
    normalizer = np.nanmax(ref_rgb)
    if normalizer == 0 or np.isnan(normalizer):
        raise ValueError("Reference composite max is zero or NaN; cannot normalize.")

    print(f"Normalizing all frames by reference max = {normalizer:.3f}")

    # 3. Helper to build and normalize one frame
    def make_frame(idx: int) -> np.ndarray:
        layers = []
        for band in bands:
            arr = ds[band].isel(time=idx).values.astype(float)
            # Normalize by the reference composite max
            arr = arr / normalizer
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            arr = np.clip(arr, 0.0, 1.0)
            # to uint8
            layers.append((arr * 255).astype(np.uint8))
        return np.stack(layers, axis=-1)
    
   
    # 4. Initialize the first frame
    pol.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)
    im = ax.imshow(make_frame(0), origin='upper', aspect='equal', extent=(left, right, bottom, top))
    if 'crs' in ds.attrs:
        pol = pol.to_crs(ds.attrs['crs'])


    list_coords_pol = [tuple(polygon.exterior.coords) for polygon in pol.geometry]
    # print(list_coords_pol[0])

    def _update(frame_idx: int):
        im.set_data(make_frame(frame_idx))
        
        return (im,)

    
    anim = FuncAnimation(fig, _update, frames=ds.sizes['time'], blit=True)
    anim.save(output_path, writer=FFMpegWriter(fps=fps))
    plt.close(fig)




def plot(stub: str):
    ds2i = pickle.load(open(f"{DS2I_DIR}/{stub}.pkl", 'rb'))
    pol = gpd.read_file(f"{SAMGEO_FILTERED_OUTPUT_VECTOR_DIR}/{stub}.gpkg")
    pol['paddock'] = range(1,len(pol)+1)
    pol['paddock'] = pol.paddock.astype('category')
    pol['color'] = 'None' # needs to be set in the gpd to achieve no colour polygon fill. 

    # Load the Fourier Transform image
    raster_path = f"{NDWI_FOURIER_GEOTIFF_DIR}/{stub}.tif"

    pf.plot_paddock_map_auto_rgb(ds2i, pol, OUT_DIR, stub)
    pf.plot_paddock_map_auto_fourier(raster_path, pol, OUT_DIR, stub)

     # Save the RGB image as a TIFF file
    output_name_rgb = os.path.join(OUT_DIR, f"{stub}_thumbs_rgb.tif")
    rgb(ds2i, 
        bands=['nbart_red', 'nbart_green', 'nbart_blue'], 
        col="time", 
        col_wrap=len(ds2i.time),
        savefig_path=output_name_rgb
    )

      # Save the veg fraction image as a TIFF file
    output_name_vegfrac = os.path.join(OUT_DIR, f'{stub}_thumbs_vegfrac.tif')
    rgb(ds2i, 
        bands=['bg', 'pv', 'npv'],
        col="time", 
        col_wrap=len(ds2i.time),
        savefig_path=output_name_vegfrac)

    # Save the time lapses of RGB and veg fract with paddocks overlaid

    custom_animator(
        ds2i,
        pol,
        ['nbart_red', 'nbart_green', 'nbart_blue'],
        f"{OUT_DIR}/{stub}+'_manpad_RGB.mp4"
    )

    # xr_animation(ds2i, 
    #             bands = ['nbart_red', 'nbart_green', 'nbart_blue'], 
    #             output_path = output_path, 
    #             show_gdf = pol, 
    #             gdf_kwargs={"edgecolor": 'white'},
    #             imshow_kwargs={"aspect": "equal"},
    #         )

    custom_animator(
        ds2i,
        pol,
        ['bg', 'pv', 'npv'],
        f"{OUT_DIR}/{stub}+'_manpad_vegfrac.mp4"
    )
    # xr_animation(ds2i, 
    #             bands = ['bg', 'pv', 'npv'], 
    #             output_path = output_path, 
    #             show_gdf = pol, 
    #             gdf_kwargs={"edgecolor": 'white'},
    #             imshow_kwargs={"aspect": "equal"},
    #             )


def test(): 
    from PaddockTSLocal.Query import get_example_query
    query = get_example_query()
    stub = query.get_stub()

    plot(stub)

if __name__ == '__main__':
    test()
