import logging
import pickle
import xarray as xr
import geopandas as gpd
from dea_tools.plotting import rgb, xr_animation
import plotting_functions as pf
import os
from PaddockTSLocal.Legend import *
import pickle

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
    xr_animation(ds2i, 
                bands = ['nbart_red', 'nbart_green', 'nbart_blue'], 
                output_path = output_path, 
                show_gdf = pol, 
                gdf_kwargs={"edgecolor": 'white'})

    output_path = f"{OUT_DIR}/{stub}+'_manpad_vegfrac.mp4"
    xr_animation(ds2i, 
                bands = ['bg', 'pv', 'npv'], 
                output_path = output_path, 
                show_gdf = pol, 
                gdf_kwargs={"edgecolor": 'white'})


def test(): 
    from PaddockTSLocal.Query import get_example_query
    query = get_example_query()
    stub = query.get_stub()

    plot(stub)

if __name__ == '__main__':
    test()
