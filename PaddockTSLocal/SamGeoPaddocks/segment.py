from PaddockTSLocal.SamGeoPaddocks.load_model import f as load_model
from PaddockTSLocal.SamGeoPaddocks.config import Config
from os.path import exists
from samgeo import SamGeo
import geopandas as gpd
import numpy as np
from geotiff import GeoTiff

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

def f(
        path_image: str,
        path_model: str,
        path_output_mask: str,
        path_output_vector: str,
        path_filtered_output_vector: str,
        min_area_ha: int = 10,
        max_area_ha: int = 1500,
        max_perim_area_ratio: int = 30
    ):
    load_model(path_model)
    model = load_model(path_model)
    model.generate(
        path_image,
        path_output_mask,
        batch=True,
        foreground=True,
        erosion_kernel=(3, 3),
        mask_multiplier=255
    )
    model.tiff_to_gpkg(path_output_mask, path_output_vector)
    filtered_gdf = filter_polygons(path_output_vector, min_area_ha, max_area_ha, max_perim_area_ratio)
    filtered_gdf.to_file(path_filtered_output_vector, driver='GPKG')

def t():
    from PaddockTSLocal.Query import Query
    from datetime import date
    from os.path import join
    from os import getcwd
    from os import makedirs

    query = Query(
        lat=-33.5040,
        lon=148.4,
        buffer=0.01,
        start_time=date(2020, 1, 1),
        end_time=date(2020, 6, 1),
        collections=['ga_s2am_ard_3', 'ga_s2bm_ard_3'],
        bands=[
            'nbart_blue',
            'nbart_green',
            'nbart_red', 
            'nbart_red_edge_1',
            'nbart_red_edge_2',
            'nbart_red_edge_3',
            'nbart_nir_1',
            'nbart_nir_2',
            'nbart_swir_2',
            'nbart_swir_3'
        ]
    )
    stub = query.get_stub()
    preseg_dir: str=join(getcwd(), 'Data', 'preseg')
    seg_dir: str = join(getcwd(), 'Data', 'seg')
    model_dir: str = join(getcwd(), 'Data', 'Samgeo', 'Model')
    makedirs(seg_dir, exist_ok=True)
    path_preseg_image = join(preseg_dir, f"{stub}.tif")
    path_output_vector = join(seg_dir, f"{stub}.gpkg")
    path_filtered_output_vector = join(seg_dir, f"{stub}_filtered.gpkg")
    path_output_mask = join(seg_dir, f"{stub}.tif")
    path_model = join(model_dir, 'sam_vit_h_4b8939.pth')
    f(path_preseg_image, path_model, path_output_mask, path_output_vector, path_filtered_output_vector)

if __name__ == '__main__':
    t()
