import gc
import glob
import os
import shutil
import time
from os.path import exists
from PaddockTS.query import Query


def run(query: Query, reload: bool = False, paddocks_filepath: str = None, skip_sam: bool = False, label_col: str | None = None):
    if skip_sam and paddocks_filepath is None:
        raise ValueError("skip_sam=True requires a valid paddocks_filepath")

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

    gpkg_path = f'{query.tmp_dir}/{query.stub}_sam_paddocks.gpkg'

    # Count total steps based on mode
    total_steps = 21  # Max steps when running both SAM and user

    total_t0 = time.time()

    # 1. Download Sentinel-2
    print(f'[1/{total_steps}] Download Sentinel-2...', flush=True)
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
    print(f'[2/{total_steps}] Compute indices...', flush=True)
    t0 = time.time()
    from PaddockTS.SpectralIndices.indices import compute_indices
    ds_sentinel2 = compute_indices(query, ds_sentinel2=ds_sentinel2)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)
    gc.collect()

    # 3. Compute fractional cover
    print(f'[3/{total_steps}] Compute fractional cover...', flush=True)
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
    print(f'[4/{total_steps}] Sentinel-2 video...', flush=True)
    t0 = time.time()
    from PaddockTS.Plotting.sentinel2_video import sentinel2_video
    sentinel2_video(query, ds_sentinel2=ds_sentinel2)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 5. Segment paddocks (SAM)
    print(f'[5/{total_steps}] Segment paddocks (SAM)...', flush=True)
    t0 = time.time()
    if skip_sam:
        print('  skipped (SAM disabled)')
    elif exists(gpkg_path):
        print('  skipped (cached)')
    else:
        from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks
        get_paddocks(query, ds_sentinel2=ds_sentinel2)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)
    gc.collect()

    # 6. S2 + paddocks video (SAM)
    print(f'[6/{total_steps}] S2 + paddocks video (SAM)...', flush=True)
    t0 = time.time()
    if skip_sam:
        print('  skipped')
    else:
        from PaddockTS.Plotting.sentinel2_paddocks_video import sentinel2_video_with_paddocks
        sentinel2_video_with_paddocks(query, paddocks_filepath=gpkg_path, ds_sentinel2=ds_sentinel2)
        print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 7. S2 + paddocks video (user)
    print(f'[7/{total_steps}] S2 + paddocks video (user)...', flush=True)
    t0 = time.time()
    if paddocks_filepath is None:
        print('  skipped')
    else:
        from PaddockTS.Plotting.sentinel2_paddocks_video import sentinel2_video_with_paddocks
        sentinel2_video_with_paddocks(query, paddocks_filepath=paddocks_filepath, ds_sentinel2=ds_sentinel2, label_col=label_col)
        print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 8. Fractional cover video
    print(f'[8/{total_steps}] Fractional cover video...', flush=True)
    t0 = time.time()
    from PaddockTS.Plotting.fractional_cover_video import fractional_cover_video
    fractional_cover_video(query, ds_fractional_cover=ds_fractional_cover)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 9. FC + paddocks video (SAM)
    print(f'[9/{total_steps}] FC + paddocks video (SAM)...', flush=True)
    t0 = time.time()
    if skip_sam:
        print('  skipped')
    else:
        from PaddockTS.Plotting.fractional_cover_paddocks_video import fractional_cover_paddocks_video
        fractional_cover_paddocks_video(query, paddocks_filepath=gpkg_path, ds_fractional_cover=ds_fractional_cover, ds_sentinel2=ds_sentinel2)
        print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 10. FC + paddocks video (user)
    print(f'[10/{total_steps}] FC + paddocks video (user)...', flush=True)
    t0 = time.time()
    if paddocks_filepath is None:
        print('  skipped')
    else:
        from PaddockTS.Plotting.fractional_cover_paddocks_video import fractional_cover_paddocks_video
        fractional_cover_paddocks_video(query, paddocks_filepath=paddocks_filepath, ds_fractional_cover=ds_fractional_cover, ds_sentinel2=ds_sentinel2, label_col=label_col)
        print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 11. Make paddock TS (SAM)
    print(f'[11/{total_steps}] Make paddock TS (SAM)...', flush=True)
    t0 = time.time()
    if skip_sam:
        print('  skipped')
        ds_paddockTS = None
    else:
        from PaddockTS.Phenology.make_paddock_time_series import make_paddock_time_series
        ds_paddockTS = make_paddock_time_series(query, ds_sentinel2=ds_sentinel2, paddocks_filepath=gpkg_path)
        print(f'  done ({time.time() - t0:.1f}s)', flush=True)
    gc.collect()

    # 12. Make paddock TS (user)
    print(f'[12/{total_steps}] Make paddock TS (user)...', flush=True)
    t0 = time.time()
    if paddocks_filepath is None:
        print('  skipped')
        ds_paddockTS_user = None
    else:
        from PaddockTS.Phenology.make_paddock_time_series import make_paddock_time_series
        ds_paddockTS_user = make_paddock_time_series(query, ds_sentinel2=ds_sentinel2, paddocks_filepath=paddocks_filepath)
        print(f'  done ({time.time() - t0:.1f}s)', flush=True)
    gc.collect()

    # 13. Make yearly paddock TS (SAM)
    print(f'[13/{total_steps}] Make yearly paddock TS (SAM)...', flush=True)
    t0 = time.time()
    if skip_sam:
        print('  skipped')
        ds_yearly = None
    else:
        from PaddockTS.Phenology.make_yearly_paddock_time_series import make_yearly_paddock_time_series
        ds_yearly = make_yearly_paddock_time_series(query, ds_paddockTS=ds_paddockTS, paddocks_filepath=gpkg_path)
        print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 14. Make yearly paddock TS (user)
    print(f'[14/{total_steps}] Make yearly paddock TS (user)...', flush=True)
    t0 = time.time()
    if paddocks_filepath is None:
        print('  skipped')
        ds_yearly_user = None
    else:
        from PaddockTS.Phenology.make_yearly_paddock_time_series import make_yearly_paddock_time_series
        ds_yearly_user = make_yearly_paddock_time_series(query, ds_paddockTS=ds_paddockTS_user, paddocks_filepath=paddocks_filepath)
        print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 15. Estimate phenology (SAM)
    print(f'[15/{total_steps}] Estimate phenology (SAM)...', flush=True)
    t0 = time.time()
    if skip_sam:
        print('  skipped')
        phenology_results = None
    else:
        from PaddockTS.Phenology.estimate_phenology import estimate_phenology
        phenology_results = estimate_phenology(query, ds_yearly=ds_yearly)
        print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 16. Estimate phenology (user)
    print(f'[16/{total_steps}] Estimate phenology (user)...', flush=True)
    t0 = time.time()
    if paddocks_filepath is None:
        print('  skipped')
        phenology_results_user = None
    else:
        from PaddockTS.Phenology.estimate_phenology import estimate_phenology
        phenology_results_user = estimate_phenology(query, ds_yearly=ds_yearly_user)
        print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 17. Calendar plot (SAM)
    print(f'[17/{total_steps}] Calendar plot (SAM)...', flush=True)
    t0 = time.time()
    if skip_sam:
        print('  skipped')
    else:
        from PaddockTS.Plotting.calendar_plot import calendar_plot
        calendar_plot(query, ds_sentinel2=ds_sentinel2, paddocks_filepath=gpkg_path)
        print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 18. Calendar plot (user)
    print(f'[18/{total_steps}] Calendar plot (user)...', flush=True)
    t0 = time.time()
    if paddocks_filepath is None:
        print('  skipped')
    else:
        from PaddockTS.Plotting.calendar_plot import calendar_plot
        calendar_plot(query, ds_sentinel2=ds_sentinel2, paddocks_filepath=paddocks_filepath, label_col=label_col)
        print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 19. Phenology plot (SAM)
    print(f'[19/{total_steps}] Phenology plot (SAM)...', flush=True)
    t0 = time.time()
    if skip_sam:
        print('  skipped')
    else:
        from PaddockTS.Plotting.phenology_plot import phenology_plot
        phenology_plot(query, phenology_results=phenology_results, ds_yearly=ds_yearly, ds_paddockTS=ds_paddockTS, paddocks_filepath=gpkg_path)
        print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 20. Phenology plot (user)
    print(f'[20/{total_steps}] Phenology plot (user)...', flush=True)
    t0 = time.time()
    if paddocks_filepath is None:
        print('  skipped')
    else:
        from PaddockTS.Plotting.phenology_plot import phenology_plot
        phenology_plot(query, phenology_results=phenology_results_user, ds_yearly=ds_yearly_user, ds_paddockTS=ds_paddockTS_user, paddocks_filepath=paddocks_filepath, label_col=label_col)
        print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 21. Make PDF report
    print(f'[21/{total_steps}] Make PDF report...', flush=True)
    t0 = time.time()
    from PaddockTS.Plotting.make_pdf import make_pdf
    make_pdf(query, paddocks_filepath=paddocks_filepath)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    print(f'\nTotal: {time.time() - total_t0:.1f}s')


if __name__ == '__main__':
    import sys
    from PaddockTS.utils import get_example_query
    run(get_example_query(), reload='--reload' in sys.argv)
