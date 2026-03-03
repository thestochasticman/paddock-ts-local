import gc
import glob
import os
import shutil
import time
from os.path import exists
from PaddockTS.query import Query


def run(query: Query, reload: bool = False):
    if reload:
        for path in [query.sentinel2_path, query.vegfrac_path]:
            if exists(path):
                shutil.rmtree(path)
        for pattern in [
            f'{query.tmp_dir}/{query.stub}_paddocks.gpkg',
            f'{query.tmp_dir}/{query.stub}_preseg.tif',
            f'{query.tmp_dir}/{query.stub}_sam_mask.tif',
            f'{query.tmp_dir}/{query.stub}_sam_raw.gpkg',
            f'{query.tmp_dir}/{query.stub}_paddockTS.zarr',
        ]:
            if exists(pattern):
                if os.path.isdir(pattern):
                    shutil.rmtree(pattern)
                else:
                    os.remove(pattern)
        for path in glob.glob(f'{query.tmp_dir}/{query.stub}_paddockTS_*.zarr'):
            shutil.rmtree(path)
        for path in glob.glob(f'{query.out_dir}/{query.stub}_calendar_*.png'):
            os.remove(path)
        for path in glob.glob(f'{query.out_dir}/{query.stub}_phenology.png'):
            os.remove(path)

    total_t0 = time.time()

    # 1. Download Sentinel-2
    print('[1/13] Download Sentinel-2...', flush=True)
    t0 = time.time()
    if not exists(query.sentinel2_path):
        from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
        ds_sentinel2 = download_sentinel2(query)
    else:
        import xarray as xr
        ds_sentinel2 = xr.open_zarr(query.sentinel2_path, chunks=None)
        print('  skipped (cached)')
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)
    gc.collect()

    # 2. Compute indices
    print('[2/13] Compute indices...', flush=True)
    t0 = time.time()
    from PaddockTS.IndicesAndVegFrac.indices import compute_indices
    ds_sentinel2 = compute_indices(query, ds_sentinel2=ds_sentinel2)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)
    gc.collect()

    # 3. Compute vegfrac
    print('[3/13] Compute vegfrac...', flush=True)
    t0 = time.time()
    if not exists(query.vegfrac_path):
        from PaddockTS.IndicesAndVegFrac.veg_frac import compute_fractional_cover
        ds_vegfrac = compute_fractional_cover(query, ds_sentinel2=ds_sentinel2)
    else:
        ds_vegfrac = xr.open_zarr(query.vegfrac_path, chunks=None)
        print('  skipped (cached)')
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)
    gc.collect()

    # 4. Sentinel-2 video
    print('[4/13] Sentinel-2 video...', flush=True)
    t0 = time.time()
    from PaddockTS.Plotting.sentinel2_video import sentinel2_video
    sentinel2_video(query, ds_sentinel2=ds_sentinel2)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 5. Segment paddocks
    print('[5/13] Segment paddocks...', flush=True)
    t0 = time.time()
    import geopandas as gpd
    gpkg_path = f'{query.tmp_dir}/{query.stub}_paddocks.gpkg'
    if exists(gpkg_path):
        paddocks = gpd.read_file(gpkg_path)
        print('  skipped (cached)')
    else:
        from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks
        paddocks = get_paddocks(query, ds_sentinel2=ds_sentinel2)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)
    gc.collect()

    # 6. Sentinel-2 + paddocks video
    print('[6/13] Sentinel-2 + paddocks video...', flush=True)
    t0 = time.time()
    from PaddockTS.Plotting.sentinel2_paddocks_video import sentinel2_video_with_paddocks
    sentinel2_video_with_paddocks(query, paddocks, ds_sentinel2=ds_sentinel2)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 7. Vegfrac video
    print('[7/13] Vegfrac video...', flush=True)
    t0 = time.time()
    from PaddockTS.Plotting.vegfrac_video import vegfrac_video
    vegfrac_video(query, ds_vegfrac=ds_vegfrac)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 8. Vegfrac + paddocks video
    print('[8/13] Vegfrac + paddocks video...', flush=True)
    t0 = time.time()
    from PaddockTS.Plotting.vegfrac_paddocks_video import vegfrac_video_with_paddocks
    vegfrac_video_with_paddocks(query, paddocks, ds_vegfrac=ds_vegfrac, ds_sentinel2=ds_sentinel2)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 9. Make paddockTS
    print('[9/13] Make paddockTS...', flush=True)
    t0 = time.time()
    from PaddockTS.PaddockTS.make_paddockTS import make_paddockTS
    ds_paddockTS = make_paddockTS(query, ds_sentinel2=ds_sentinel2, paddocks=paddocks)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)
    gc.collect()

    # 10. Make yearly paddockTS
    print('[10/13] Make yearly paddockTS...', flush=True)
    t0 = time.time()
    from PaddockTS.PaddockTS.make_yearly_paddockTS import make_yearly_paddockTS
    ds_yearly = make_yearly_paddockTS(query, ds_paddockTS=ds_paddockTS)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 11. Estimate phenology
    print('[11/13] Estimate phenology...', flush=True)
    t0 = time.time()
    from PaddockTS.Phenology.estimate_phenology import estimate_phenology
    phenology_results = estimate_phenology(query, ds_yearly=ds_yearly)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 12. Calendar plot
    print('[12/13] Calendar plot...', flush=True)
    t0 = time.time()
    from PaddockTS.Plotting.calendar_plot import calendar_plot
    calendar_plot(query, ds_sentinel2=ds_sentinel2, paddocks=paddocks)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 13. Phenology plot
    print('[13/13] Phenology plot...', flush=True)
    t0 = time.time()
    from PaddockTS.Plotting.phenology_plot import phenology_plot
    phenology_plot(query, phenology_results=phenology_results, ds_yearly=ds_yearly, ds_paddockTS=ds_paddockTS)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    print(f'\nTotal: {time.time() - total_t0:.1f}s')


if __name__ == '__main__':
    import sys
    from PaddockTS.utils import get_example_query
    run(get_example_query(), reload='--reload' in sys.argv)
