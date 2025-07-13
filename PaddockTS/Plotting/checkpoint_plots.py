import logging
import pickle
import xarray as xr
import geopandas as gpd
from dea_tools.plotting import rgb, xr_animation

import PaddockTS.Plotting.plotting_functions as pf
import os
from PaddockTS.legend import *
import pickle
from shapely.geometry import Polygon
import cv2
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

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
    output_name_vegfrac = os.path.join(OUT_DIR, f"{stub}_thumbs_vegfrac.tif")
    rgb(ds2i, 
        bands=['bg', 'pv', 'npv'],
        col="time", 
        col_wrap=len(ds2i.time),
        savefig_path=output_name_vegfrac)
    

    pf.animate_paddock_map_auto_rgb(
        ds2i,
        pol,
        ['nbart_red', 'nbart_green', 'nbart_blue'],
        out_dir=OUT_DIR,
        stub=stub

    )
    
    pf.animate_paddock_map_auto_manpad_vegfrac(
        ds2i,
        pol,
        ['bg', 'pv', 'npv'],
        out_dir=OUT_DIR,
        stub=stub
    )


def test(): 
    from PaddockTS.query import get_example_query, Query
    from datetime import date
    query = get_example_query()

    query = Query(
        lat=-33.5040,
        lon=148.4,
        buffer=0.01,
        start_time=date(2020, 1, 1),
        end_time=date(2020, 6, 1),
        collections=['ga_s2am_ard_3', 'ga_s2bm_ard_3'],
        bands=[
            'nbart_blue', 'nbart_green', 'nbart_red', 'nbart_red_edge_1', 'nbart_red_edge_2', 'nbart_red_edge_3', 'nbart_nir_1', 'nbart_nir_2', 'nbart_swir_2', 'nbart_swir_3'
        ]
    )
    stub = query.get_stub()
    plot(stub)

if __name__ == '__main__':
    test()
