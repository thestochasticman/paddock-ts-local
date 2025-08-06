
from PaddockTS.query import Query
from os.path import dirname
from os.path import dirname
from os.path import exists
from samgeo import SamGeo
from os import makedirs
import geopandas as gpd
import numpy as np
import torch
import wget

torch.set_default_dtype(torch.float32)

def download_weights(path: str)->None:
    '''
    Download the SAM model weights
    '''
    makedirs(dirname(path), exist_ok=True)
    url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
    wget.download(url, out=path)
    

def load_model(path: str, device='cpu')->SamGeo:
    '''
    Load the SamGeo model from checkpoint at 'path', on the given device.
    Downloads weights if missing.
    '''
    if not exists(path): download_weights(path)
    return SamGeo(model_type='vit_h', checkpoint=path, device=device)

def filter_polygons(vector, min_area_ha, max_area_ha, max_perim_area_ratio):
    '''
    Read a vector file, compute area and perimeter ratios, and filter geometries.
    - Filters by min/max area in hectares.
    - Filters by maximum perimeter-to-area ratio.
    '''
    pol = gpd.read_file(vector).drop(labels = 'value', axis = 1)
    pol['area_ha'] = pol.area/1000
    pol['log_area_ha'] = np.log10(pol['area_ha'])
    pol['perim-area'] = pol.length/pol['area_ha']
    pol_filt = pol[
        (pol['area_ha'] >= min_area_ha) &
        (pol['area_ha'] <= max_area_ha) &
        (pol['perim-area'] <= max_perim_area_ratio)
    ]
    return pol_filt

def segment(
        query: Query,
        min_area_ha: int = 10,
        max_area_ha: int = 1500,
        max_perim_area_ratio: int = 30,
        device='cpu'
):
    '''
    1. Loads pre-segmentation GeoTIFF based on NDWI-Fourier features.
    2. Runs SAMgeo to generate a mask.
    3. Converts mask to vector format (GeoPackage).
    4. Filters polygons by size/shape and saves filtered output.
    '''
    path_preseg_image = query.path_preseg_tif
    path_output_mask = f"{query.stub_tmp_dir}/mask.tif"
    path_output_vector = f"{query.stub_tmp_dir}/vector.gpkg"
    path_filtered_output_vector = query.path_polygons
    samgeo_model_path = f"{query.stub_tmp_dir}/sam_vit_h_4b8939.pth"
    model = load_model(samgeo_model_path, device)
    model.generate(
        path_preseg_image,
        path_output_mask,
        batch=True,
        foreground=True,
        erosion_kernel=(3, 3),
        mask_multiplier=255
    )
    model.tiff_to_gpkg(path_output_mask, path_output_vector)
    filtered_gdf = filter_polygons(path_output_vector, min_area_ha, max_area_ha, max_perim_area_ratio)
    filtered_gdf.to_file(path_filtered_output_vector, driver='GPKG')
    
def test():
    from PaddockTS.query import get_example_query
    query = get_example_query()
    segment(query, device='cpu')

if __name__ == '__main__':
    test()