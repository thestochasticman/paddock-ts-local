from PaddockTSLocal.legend import SAMGEO_FILTERED_OUTPUT_VECTOR_DIR
from PaddockTSLocal.legend import SAMGEO_OUTPUT_VECTOR_DIR
from PaddockTSLocal.legend import NDWI_FOURIER_GEOTIFF_DIR
from PaddockTSLocal.legend import SAMGEO_OUTPUT_MASK_DIR
from PaddockTSLocal.legend import SAMGEO_MODEL_PATH
from geotiff import GeoTiff
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
    makedirs(dirname(path), exist_ok=True)
    url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
    wget.download(url, out=path)

def load_model(path: str, device='cpu')->SamGeo:
    if not exists(path): download_weights(path)
    return SamGeo(model_type='vit_h', checkpoint=path, device=device)

def filter_polygons(vector, min_area_ha, max_area_ha, max_perim_area_ratio):
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
        stub: str,
        min_area_ha: int = 10,
        max_area_ha: int = 1500,
        max_perim_area_ratio: int = 30,
        device='cpu'
):
    path_preseg_image = f"{NDWI_FOURIER_GEOTIFF_DIR}/{stub}.tif"
    path_output_mask = f"{SAMGEO_OUTPUT_MASK_DIR}/{stub}.tif"
    path_output_vector = f"{SAMGEO_OUTPUT_VECTOR_DIR}.gpkg"
    path_filtered_output_vector = f"{SAMGEO_FILTERED_OUTPUT_VECTOR_DIR}/{stub}.gpkg"
    model = load_model(SAMGEO_MODEL_PATH, device)
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
    from PaddockTSLocal.query import get_example_query

    query = get_example_query()
    stub = query.get_stub()
    segment(stub, device='cpu')

if __name__ == '__main__':
    test()