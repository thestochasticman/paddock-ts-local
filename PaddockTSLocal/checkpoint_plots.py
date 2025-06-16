import logging
import pickle
import xarray as xr
import geopandas as gpd
from dea_tools.plotting import rgb, xr_animation
import plotting_functions as pf
import os
from PaddockTSLocal.Legend import *
import pickle

import cv2
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def custom_animator(
    ds: xr.Dataset,
    bands: list[str],
    output_path: str,
    width_pixels: int = 600,
    fps: int = 10,
    dpi: int = 100,
    scale_factor: float = 10000.0
):
    """
    Animate an xarray Dataset along its 'time' dimension by:
      1. Dividing raw band values by scale_factor
      2. Replacing NaNs with 0
      3. Clipping to [0…1]
      4. Converting to uint8 [0…255]
      5. Rendering with square pixels (aspect='equal')
    """
    # 1. Compute figure size & aspect ratio
    ny, nx = ds.sizes['y'], ds.sizes['x']
    aspect = ny / nx
    fig_w, fig_h = width_pixels / dpi, (width_pixels * aspect) / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.axis('off')

    # 2. Helper to build and normalize one frame
    def make_frame(idx: int) -> np.ndarray:
        layers = []
        for b in bands:
            arr = ds[b].isel(time=idx).values.astype(float) / scale_factor
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            arr = np.clip(arr, 0.0, 1.0)
            layers.append((arr * 255).astype(np.uint8))
        return np.stack(layers, axis=-1)

    # 3. Diagnostic print: show the range after scaling (0–255)
    all_vals = []
    for b in bands:
        vals = (ds[b].values.astype(float) / scale_factor).flatten()
        all_vals.append(vals[np.isfinite(vals)])
    all_vals = np.concatenate(all_vals)
    print(f"Post-scale (uint8) data range: [{int(all_vals.min()*255)} … {int(all_vals.max()*255)}]")

    # 4. Initialize the first frame
    im = ax.imshow(make_frame(0), origin='upper', aspect='equal')

    # 5. Update function for FuncAnimation
    def _update(frame_idx: int):
        im.set_data(make_frame(frame_idx))
        return (im,)

    # 6. Build and save the animation
    anim = FuncAnimation(fig, _update, frames=ds.sizes['time'], blit=True)
    anim.save(output_path, writer=FFMpegWriter(fps=fps))
    plt.close(fig)

def custom_animator(
    ds: xr.Dataset,
    bands: list[str],
    output_path: str,
    width_pixels: int = 600,
    fps: int = 10,
    dpi: int = 100,
    scale_factor: float = 10000.0,
    clip_limit: float = 2.0,          # for CLAHE
    tile_grid_size: tuple = (8, 8)    # for CLAHE
):
    # 1. Figure size & aspect
    ny, nx = ds.sizes['y'], ds.sizes['x']
    aspect = ny / nx
    fig_w, fig_h = width_pixels/dpi, (width_pixels*aspect)/dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.axis('off')

    # 2. Setup CLAHE once
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # 3. Frame builder with scaling+CLAHE
    def make_frame(idx: int) -> np.ndarray:
        # a) Load & normalize to [0…255] uint8
        layers = []
        for b in bands:
            arr = ds[b].isel(time=idx).values.astype(float) / scale_factor
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            arr = np.clip(arr, 0.0, 1.0)
            layers.append((arr * 255).astype(np.uint8))
        rgb = np.stack(layers, axis=-1)  # shape (y,x,3), RGB

        # b) Convert RGB→LAB
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b_ = cv2.split(lab)

        # c) Apply CLAHE on L channel
        l_eq = clahe.apply(l)

        # d) Merge & convert back to RGB
        lab_eq = cv2.merge((l_eq, a, b_))
        rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return rgb_eq

    # 4. Init & animate
    im = ax.imshow(make_frame(0), origin='upper', aspect='equal')
    def _update(i):
        im.set_data(make_frame(i))
        return (im,)

    anim = FuncAnimation(fig, _update, frames=ds.sizes['time'], blit=True)
    anim.save(output_path, writer=FFMpegWriter(fps=fps))
    plt.close(fig)

def custom_animator(
    ds: xr.Dataset,
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
    im = ax.imshow(make_frame(0), origin='upper', aspect='equal')

    # 5. Update function for FuncAnimation
    def _update(frame_idx: int):
        im.set_data(make_frame(frame_idx))
        return (im,)

    # 6. Build and save the animation
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
    output_path = f"{OUT_DIR}/{stub}+'_manpad_RGB.mp4"
    custom_animator(ds2i, ['nbart_red', 'nbart_green', 'nbart_blue'], output_path=output_path, width_pixels=215)
    # xr_animation(ds2i, 
    #             bands = ['nbart_red', 'nbart_green', 'nbart_blue'], 
    #             output_path = output_path, 
    #             show_gdf = pol, 
    #             gdf_kwargs={"edgecolor": 'white'},
    #             imshow_kwargs={"aspect": "equal"},
    #         )

    # output_path = f"{OUT_DIR}/{stub}+'_manpad_vegfrac.mp4"
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
