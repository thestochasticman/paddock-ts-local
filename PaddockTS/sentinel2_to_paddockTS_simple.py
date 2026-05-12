import gc
import glob
import os
import shutil
import time
from os.path import exists
from PaddockTS.query import Query


def run(query: Query, reload: bool = False):
    if reload:
        for path in [query.sentinel2_path, query.fractional_cover_path]:
            if exists(path):
                shutil.rmtree(path)
        for pattern in [
            f'{query.tmp_dir}/{query.stub}_sam_paddocks.gpkg',
            f'{query.tmp_dir}/{query.stub}_preseg.tif',
            f'{query.tmp_dir}/{query.stub}_sam_mask.tif',
            f'{query.tmp_dir}/{query.stub}_sam_raw.gpkg',
            f'{query.tmp_dir}/{query.stub}_paddocks_timeseries.zarr',
        ]:
            if exists(pattern):
                if os.path.isdir(pattern):
                    shutil.rmtree(pattern)
                else:
                    os.remove(pattern)
        for path in glob.glob(f'{query.tmp_dir}/{query.stub}_paddocks_timeseries_*.zarr'):
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
    from PaddockTS.SpectralIndices.indices import compute_indices
    ds_sentinel2 = compute_indices(query, ds_sentinel2=ds_sentinel2)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)
    gc.collect()

    # 3. Compute fractional cover
    print('[3/13] Compute fractional cover...', flush=True)
    t0 = time.time()
    if not exists(query.fractional_cover_path):
        from PaddockTS.FractionalCover.compute_fractional_cover import compute_fractional_cover
        ds_fractional_cover = compute_fractional_cover(query, ds_sentinel2=ds_sentinel2)
    else:
        ds_fractional_cover = xr.open_zarr(query.fractional_cover_path, chunks=None)
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
    gpkg_path = f'{query.tmp_dir}/{query.stub}_sam_paddocks.gpkg'
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

    # 7. Fractional cover video
    print('[7/13] Fractional cover video...', flush=True)
    t0 = time.time()
    from PaddockTS.Plotting.fractional_cover_video import fractional_cover_video
    fractional_cover_video(query, ds_fractional_cover=ds_fractional_cover)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 8. Fractional cover + paddocks video
    print('[8/13] Fractional cover + paddocks video...', flush=True)
    t0 = time.time()
    from PaddockTS.Plotting.fractional_cover_paddocks_video import fractional_cover_paddocks_video
    fractional_cover_paddocks_video(query, paddocks, ds_fractional_cover=ds_fractional_cover, ds_sentinel2=ds_sentinel2)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 9. Make paddock time series
    print('[9/13] Make paddock time series...', flush=True)
    t0 = time.time()
    from PaddockTS.Phenology.make_paddock_time_series import make_paddock_time_series
    gpkg_path = f'{query.tmp_dir}/{query.stub}_sam_paddocks.gpkg'
    ds_paddockTS = make_paddock_time_series(query, ds_sentinel2=ds_sentinel2, paddocks_filepath=gpkg_path)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)
    gc.collect()

    # 10. Make yearly paddock time series
    print('[10/13] Make yearly paddock time series...', flush=True)
    t0 = time.time()
    from PaddockTS.Phenology.make_yearly_paddock_time_series import make_yearly_paddock_time_series
    ds_yearly = make_yearly_paddock_time_series(query, ds_paddockTS=ds_paddockTS, paddocks_filepath=gpkg_path)
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
