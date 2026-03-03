import gc
import glob
import io
import os
import shutil
import time
from contextlib import redirect_stdout
from os.path import exists
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console, Group
from PaddockTS.query import Query

_console = Console(stderr=True)

STEPS = [
    'Download Sentinel-2',
    'Compute indices',
    'Compute vegfrac',
    'Sentinel-2 video',
    'Segment paddocks',
    'Sentinel-2 + paddocks video',
    'Vegfrac video',
    'Vegfrac + paddocks video',
    'Make paddockTS',
    'Make yearly paddockTS',
    'Estimate phenology',
    'Calendar plot',
    'Phenology plot',
]


def _make_table(statuses, times):
    table = Table(title='Sentinel-2 to PaddockTS Pipeline')
    table.add_column('#', style='dim', width=3)
    table.add_column('Step', width=30)
    table.add_column('Status', width=12)
    table.add_column('Time', width=10)
    for i, name in enumerate(STEPS):
        status = statuses[i]
        elapsed = times[i]
        if status == 'done':
            style, symbol = 'green', '[green]done[/green]'
        elif status == 'running':
            style, symbol = 'yellow', '[yellow]running...[/yellow]'
        elif status == 'skipped':
            style, symbol = 'dim', '[dim]skipped[/dim]'
        else:
            style, symbol = 'dim', '[dim]pending[/dim]'
        time_str = f'{elapsed:.1f}s' if elapsed is not None else ''
        table.add_row(str(i + 1), name, symbol, time_str)
    return table


def run(query: Query, reload: bool = False):
    import xarray as xr

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

    statuses = ['pending'] * len(STEPS)
    times = [None] * len(STEPS)

    ds_sentinel2 = None
    ds_vegfrac = None
    paddocks = None
    ds_paddockTS = None
    ds_yearly = None
    phenology_results = None

    progress = Progress(
        TextColumn('[bold blue]{task.description}'),
        BarColumn(),
        TextColumn('{task.completed}/{task.total}'),
        TimeElapsedColumn(),
        console=_console,
    )
    task_id = progress.add_task('Pipeline', total=len(STEPS))

    os.makedirs(query.tmp_dir, exist_ok=True)
    log = open(f'{query.tmp_dir}/{query.stub}_sentinel2_to_paddockTS.log', 'w')

    with Live(Group(_make_table(statuses, times), progress), console=_console, refresh_per_second=4) as live:
        for i in range(len(STEPS)):
            statuses[i] = 'running'
            live.update(Group(_make_table(statuses, times), progress))

            # Clean up before segmentation step
            if i == 4:
                gc.collect()
                for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
                    os.environ.pop(var, None)
                try:
                    import torch
                    torch.set_num_threads(os.cpu_count() or 4)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            t0 = time.time()
            try:
                # Segmentation needs live console for SAM progress
                if i == 4:
                    live.stop()

                with redirect_stdout(log) if i != 4 else open(os.devnull) as _:
                    if i == 0:
                        # Download Sentinel-2
                        if not exists(query.sentinel2_path):
                            from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
                            ds_sentinel2 = download_sentinel2(query)
                        else:
                            ds_sentinel2 = xr.open_zarr(query.sentinel2_path, chunks=None)

                    elif i == 1:
                        # Compute indices
                        from PaddockTS.IndicesAndVegFrac.indices import compute_indices
                        ds_sentinel2 = compute_indices(query, ds_sentinel2=ds_sentinel2)

                    elif i == 2:
                        # Compute vegfrac
                        from PaddockTS.IndicesAndVegFrac.veg_frac import compute_fractional_cover
                        ds_vegfrac = compute_fractional_cover(query, ds_sentinel2=ds_sentinel2)

                    elif i == 3:
                        # Sentinel-2 video
                        from PaddockTS.Plotting.sentinel2_video import sentinel2_video
                        sentinel2_video(query, ds_sentinel2=ds_sentinel2)

                    elif i == 4:
                        # Segment paddocks
                        import geopandas as gpd
                        gpkg_path = f'{query.tmp_dir}/{query.stub}_paddocks.gpkg'
                        if exists(gpkg_path):
                            paddocks = gpd.read_file(gpkg_path)
                        else:
                            from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks
                            paddocks = get_paddocks(query, ds_sentinel2=ds_sentinel2)

                    elif i == 5:
                        # Sentinel-2 + paddocks video
                        from PaddockTS.Plotting.sentinel2_paddocks_video import sentinel2_video_with_paddocks
                        sentinel2_video_with_paddocks(query, paddocks, ds_sentinel2=ds_sentinel2)

                    elif i == 6:
                        # Vegfrac video
                        from PaddockTS.Plotting.vegfrac_video import vegfrac_video
                        vegfrac_video(query, ds_vegfrac=ds_vegfrac)

                    elif i == 7:
                        # Vegfrac + paddocks video
                        from PaddockTS.Plotting.vegfrac_paddocks_video import vegfrac_video_with_paddocks
                        vegfrac_video_with_paddocks(query, paddocks, ds_vegfrac=ds_vegfrac, ds_sentinel2=ds_sentinel2)

                    elif i == 8:
                        # Make paddockTS
                        from PaddockTS.PaddockTS.make_paddockTS import make_paddockTS
                        ds_paddockTS = make_paddockTS(query, ds_sentinel2=ds_sentinel2, paddocks=paddocks)

                    elif i == 9:
                        # Make yearly paddockTS
                        from PaddockTS.PaddockTS.make_yearly_paddockTS import make_yearly_paddockTS
                        ds_yearly = make_yearly_paddockTS(query, ds_paddockTS=ds_paddockTS)

                    elif i == 10:
                        # Estimate phenology
                        from PaddockTS.Phenology.estimate_phenology import estimate_phenology
                        phenology_results = estimate_phenology(query, ds_yearly=ds_yearly)

                    elif i == 11:
                        # Calendar plot
                        from PaddockTS.Plotting.calendar_plot import calendar_plot
                        calendar_plot(query, ds_sentinel2=ds_sentinel2, paddocks=paddocks)

                    elif i == 12:
                        # Phenology plot
                        from PaddockTS.Plotting.phenology_plot import phenology_plot
                        phenology_plot(query, phenology_results=phenology_results, ds_yearly=ds_yearly, ds_paddockTS=ds_paddockTS)

                if i == 4:
                    live.start()

                statuses[i] = 'done'
            except Exception:
                if i == 4:
                    live.start()
                statuses[i] = '[red]error[/red]'
                times[i] = time.time() - t0
                live.update(Group(_make_table(statuses, times), progress))
                log.close()
                raise

            times[i] = time.time() - t0
            progress.update(task_id, completed=i + 1)
            live.update(Group(_make_table(statuses, times), progress))

    log.close()


if __name__ == '__main__':
    import sys
    from PaddockTS.utils import get_example_query
    run(get_example_query(), reload='--reload' in sys.argv)
