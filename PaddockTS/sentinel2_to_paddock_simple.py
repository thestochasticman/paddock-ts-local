import gc
import os
import time
import shutil
from os.path import exists
from PaddockTS.query import Query


def run(query: Query, reload: bool = False):
    if reload:
        for path in [query.sentinel2_path, query.vegfrac_path]:
            if exists(path):
                shutil.rmtree(path)
        for suffix in ['_paddocks.gpkg', '_preseg.tif', '_sam_mask.tif', '_sam_raw.gpkg']:
            path = f'{query.tmp_dir}/{query.stub}{suffix}'
            if exists(path):
                os.remove(path)

    total_t0 = time.time()

    # 1. Download Sentinel-2
    print('[1/8] Download Sentinel-2...', flush=True)
    t0 = time.time()
    if not exists(query.sentinel2_path):
        from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
        download_sentinel2(query)
    else:
        print('  skipped (cached)')
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)
    gc.collect()

    # 2. Compute indices
    print('[2/8] Compute indices...', flush=True)
    t0 = time.time()
    from PaddockTS.IndicesAndVegFrac.indices import compute_indices
    compute_indices(query)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)
    gc.collect()

    # 3. Compute vegfrac
    print('[3/8] Compute vegfrac...', flush=True)
    t0 = time.time()
    if not exists(query.vegfrac_path):
        from PaddockTS.IndicesAndVegFrac.veg_frac import compute_fractional_cover
        compute_fractional_cover(query)
    else:
        print('  skipped (cached)')
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)
    gc.collect()

    # Reset thread env vars that Dask sets to 1
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ.pop(var, None)
    try:
        import torch
        torch.set_num_threads(os.cpu_count() or 4)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # 4. Segment paddocks
    print('[4/8] Segment paddocks...', flush=True)
    t0 = time.time()
    import geopandas as gpd
    gpkg_path = f'{query.tmp_dir}/{query.stub}_paddocks.gpkg'
    if exists(gpkg_path):
        paddocks = gpd.read_file(gpkg_path)
        print('  skipped (cached)')
    else:
        from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks
        paddocks = get_paddocks(query)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 5. Sentinel-2 video
    print('[5/8] Sentinel-2 video...', flush=True)
    t0 = time.time()
    from PaddockTS.Plotting.sentinel2_video import sentinel2_video
    sentinel2_video(query)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 6. Sentinel-2 + paddocks video
    print('[6/8] Sentinel-2 + paddocks video...', flush=True)
    t0 = time.time()
    from PaddockTS.Plotting.sentinel2_paddocks_video import sentinel2_video_with_paddocks
    sentinel2_video_with_paddocks(query, paddocks)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 7. Vegfrac video
    print('[7/8] Vegfrac video...', flush=True)
    t0 = time.time()
    from PaddockTS.Plotting.vegfrac_video import vegfrac_video
    vegfrac_video(query)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    # 8. Vegfrac + paddocks video
    print('[8/8] Vegfrac + paddocks video...', flush=True)
    t0 = time.time()
    from PaddockTS.Plotting.vegfrac_paddocks_video import vegfrac_video_with_paddocks
    vegfrac_video_with_paddocks(query, paddocks)
    print(f'  done ({time.time() - t0:.1f}s)', flush=True)

    print(f'\nTotal: {time.time() - total_t0:.1f}s')


if __name__ == '__main__':
    import sys
    from PaddockTS.utils import get_example_query
    run(get_example_query(), reload='--reload' in sys.argv)
