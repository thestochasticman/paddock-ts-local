import gc
import glob
import io
import os
import shutil
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from os.path import exists
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console, Group
from PaddockTS.query import Query

# Use the original stderr (sys.__stderr__) so Live keeps drawing to the terminal
# even when steps redirect stderr into log files (e.g. samgeo's tqdm output).
_console = Console(file=sys.__stderr__, force_terminal=True)

STEPS = [
    'Download Sentinel-2',
    'Compute indices',
    'Compute fractional cover',
    'Sentinel-2 video',
    'Segment paddocks',
    'Sentinel-2 + paddocks video',
    'Fractional cover video',
    'Fractional cover + paddocks video',
    'Make paddock time series',
    'Make yearly paddock time series',
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
        for path in [query.sentinel2_path, query.fractional_cover_path]:
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
    ds_fractional_cover = None
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
                with redirect_stdout(log), redirect_stderr(log):
                    if i == 0:
                        # Download Sentinel-2
                        if not exists(query.sentinel2_path):
                            from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
                            ds_sentinel2 = download_sentinel2(query)
                        else:
                            ds_sentinel2 = xr.open_zarr(query.sentinel2_path, chunks=None)

                    elif i == 1:
                        # Compute indices
                        from PaddockTS.SpectralIndices.indices import compute_indices
                        ds_sentinel2 = compute_indices(query, ds_sentinel2=ds_sentinel2)

                    elif i == 2:
                        # Compute fractional cover
                        from PaddockTS.FractionalCover.compute_fractional_cover import compute_fractional_cover
                        ds_fractional_cover = compute_fractional_cover(query, ds_sentinel2=ds_sentinel2)

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
                        # Fractional cover video
                        from PaddockTS.Plotting.fractional_cover_video import fractional_cover_video
                        fractional_cover_video(query, ds_fractional_cover=ds_fractional_cover)

                    elif i == 7:
                        # Fractional cover + paddocks video
                        from PaddockTS.Plotting.fractional_cover_paddocks_video import fractional_cover_paddocks_video
                        fractional_cover_paddocks_video(query, paddocks, ds_fractional_cover=ds_fractional_cover, ds_sentinel2=ds_sentinel2)

                    elif i == 8:
                        # Make paddock time series
                        from PaddockTS.PaddockTimeSeries.make_paddock_time_series import make_paddock_time_series
                        ds_paddockTS = make_paddock_time_series(query, ds_sentinel2=ds_sentinel2, paddocks=paddocks)

                    elif i == 9:
                        # Make yearly paddock time series
                        from PaddockTS.PaddockTimeSeries.make_yearly_paddock_time_series import make_yearly_paddock_time_series
                        ds_yearly = make_yearly_paddock_time_series(query, ds_paddockTS=ds_paddockTS)

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

                statuses[i] = 'done'
            except Exception:
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
