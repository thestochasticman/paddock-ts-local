import torch

from geotiff import GeoTiff
from os.path import dirname
from os.path import exists
# from samgeo import SamGeo
from PaddockTSLocal.CustomSamGeo import SamGeo
import xarray as xr
from os import makedirs
import geopandas as gpd
import numpy as np
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
        path_geotiff: str,
        path_model: str,
        path_output_mask: str,
        path_output_vector: str,
        path_filtered_output_vector: str,
        min_area_ha: int = 10,
        max_area_ha: int = 1500,
        max_perim_area_ratio: int = 30,
        device='cpu'
):
    model = load_model(path_model, device)
    model.generate(
        path_geotiff,
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
    from PaddockTSLocal.Query import get_example_query
    from os.path import join
    from os import getcwd

    query = get_example_query()
    stub = query.get_stub()
    preseg_dir: str=join(getcwd(), 'Data', 'ndwi_tiff')
    seg_dir: str = join(getcwd(), 'Data', 'seg')
    model_dir: str = join(getcwd(), 'Data', 'Samgeo', 'Model')
    makedirs(seg_dir, exist_ok=True)
    path_preseg_image = join(preseg_dir, f"{stub}.tif")
    path_output_vector = join(seg_dir, f"{stub}.gpkg")
    path_filtered_output_vector = join(seg_dir, f"{stub}_filtered.gpkg")
    path_output_mask = join(seg_dir, f"{stub}.tif")
    path_model = join(model_dir, 'sam_vit_h_4b8939.pth')
    segment(path_preseg_image, path_model, path_output_mask, path_output_vector, path_filtered_output_vector, device='cpu')

if __name__ == '__main__':
    test()