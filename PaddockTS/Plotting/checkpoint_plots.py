import logging
import pickle
import xarray as xr
import geopandas as gpd
from dea_tools.plotting import rgb, xr_animation

import PaddockTS.Plotting.plotting_functions as pf
import os
import pickle
from shapely.geometry import Polygon
import cv2
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from PaddockTS.query import Query

def plot(query: Query):
    path_ds2i = f"{query.stub_tmp_dir}/DS2I.pkl"
    print(path_ds2i)
    ds2i = pickle.load(open(path_ds2i, 'rb'))
    pol = gpd.read_file(query.path_polygons)
    pol['paddock'] = range(1,len(pol)+1)
    pol['paddock'] = pol.paddock.astype('category')
    pol['color'] = 'None' # needs to be set in the gpd to achieve no colour polygon fill. 

    # Load the Fourier Transform image
    raster_path = query.path_preseg_tif
    out_dir = query.dir_checkpoint_plots
    pf.plot_paddock_map_auto_rgb(ds2i, pol, out_dir, query.stub)
    pf.plot_paddock_map_auto_fourier(raster_path, pol, out_dir, query.stub)

     # Save the RGB image as a TIFF file
    # output_name_rgb = os.path.join(out_dir, f"{query.stub}_thumbs_rgb.tif")
    output_name_rgb=f"{out_dir}/{query.stub}_thumbs_rgb.png"
    rgb(ds2i, 
        bands=['nbart_red', 'nbart_green', 'nbart_blue'], 
        col="time", 
        col_wrap=len(ds2i.time),
        savefig_path=output_name_rgb
    )

      # Save the veg fraction image as a TIFF file
    # output_name_vegfrac = os.path.join(out_dir, f"{query.stub}_thumbs_vegfrac.tif")
    output_name_vegfrac = f"{out_dir}/{query.stub}_thumbs_vegfrac.png"
    rgb(ds2i, 
        bands=['bg', 'pv', 'npv'],
        col="time", 
        col_wrap=len(ds2i.time),
        savefig_path=output_name_vegfrac)
    

    pf.animate_paddock_map_auto_rgb(
        ds2i,
        pol,
        ['nbart_red', 'nbart_green', 'nbart_blue'],
        out_dir=out_dir,
        stub=query.stub
    )
    
    pf.animate_paddock_map_auto_manpad_vegfrac(
        ds2i,
        pol,
        ['bg', 'pv', 'npv'],
        out_dir=out_dir,
        stub=query.stub
    )


def test(): 
    from PaddockTS.query import get_example_query, Query
    from datetime import date
    query = get_example_query()

    plot(query)

if __name__ == '__main__':
    test()
