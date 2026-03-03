import gc
import os
import shutil
import time
import threading
from contextlib import redirect_stdout
from os.path import exists
from rich.live import Live
from rich.table import Table
from rich.columns import Columns
from rich.console import Console
from PaddockTS.query import Query

_console = Console(stderr=True)

ENV_STEPS = [
    'Download terrain',
    'Download OzWALD daily',
    'Download SILO',
    'Download SLGA soils',
    'OzWALD plot',
    'SILO plot',
    'Terrain plot',
]

S2_STEPS = [
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


def _make_table(title, steps, statuses, times):
    table = Table(title=title)
    table.add_column('#', style='dim', width=3)
    table.add_column('Step', width=30)
    table.add_column('Status', width=12)
    table.add_column('Time', width=10)
    for i, name in enumerate(steps):
        status = statuses[i]
        elapsed = times[i]
        if status == 'done':
            symbol = '[green]done[/green]'
        elif status == 'running':
            symbol = '[yellow]running...[/yellow]'
        elif status == 'waiting':
            symbol = '[cyan]waiting...[/cyan]'
        elif status == 'error':
            symbol = '[red]error[/red]'
        else:
            symbol = '[dim]pending[/dim]'
        time_str = f'{elapsed:.1f}s' if elapsed is not None else ''
        table.add_row(str(i + 1), name, symbol, time_str)
    return table


def _run_env_steps(query, statuses, times):
    os.makedirs(query.tmp_dir, exist_ok=True)
    log = open(f'{query.tmp_dir}/{query.stub}_environmental.log', 'w')
    for i in range(len(ENV_STEPS)):
        statuses[i] = 'running'
        t0 = time.time()
        try:
            with redirect_stdout(log):
                if i == 0:
                    from PaddockTS.Environmental.TerrainTiles.download_terrain_tiles import download_terrain
                    download_terrain(query)
                elif i == 1:
                    from PaddockTS.Environmental.OzWALD.download_ozwald_daily import download_ozwald_daily
                    download_ozwald_daily(query)
                elif i == 2:
                    from PaddockTS.Environmental.SILO.download_silo import download_silo
                    download_silo(query)
                elif i == 3:
                    from PaddockTS.Environmental.SLGASoils.download_slgasoils import download_slga_soils
                    download_slga_soils(query)
                elif i == 4:
                    from PaddockTS.Plotting.ozwald_plot import ozwald_daily_plot
                    ozwald_daily_plot(query)
                elif i == 5:
                    from PaddockTS.Plotting.silo_plot import silo_plot
                    silo_plot(query)
                elif i == 6:
                    statuses[i] = 'waiting'
                    while not exists(query.sentinel2_path):
                        time.sleep(1)
                    statuses[i] = 'running'
                    from PaddockTS.Plotting.terrain_tiles_plot import terrain_tiles_plot
                    terrain_tiles_plot(query)
            statuses[i] = 'done'
        except Exception:
            statuses[i] = 'error'
            times[i] = time.time() - t0
            log.close()
            raise
        times[i] = time.time() - t0
    log.close()


def _run_s2_steps(query, statuses, times):
    import xarray as xr

    ds_sentinel2 = None
    ds_vegfrac = None
    paddocks = None
    ds_paddockTS = None
    ds_yearly = None
    phenology_results = None

    os.makedirs(query.tmp_dir, exist_ok=True)
    log = open(f'{query.tmp_dir}/{query.stub}_sentinel2_to_paddockTS.log', 'w')
    for i in range(len(S2_STEPS)):
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

        statuses[i] = 'running'
        t0 = time.time()
        try:
            with redirect_stdout(log):
                if i == 0:
                    if not exists(query.sentinel2_path):
                        from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
                        ds_sentinel2 = download_sentinel2(query)
                    else:
                        ds_sentinel2 = xr.open_zarr(query.sentinel2_path, chunks=None)
                elif i == 1:
                    from PaddockTS.IndicesAndVegFrac.indices import compute_indices
                    ds_sentinel2 = compute_indices(query, ds_sentinel2=ds_sentinel2)
                elif i == 2:
                    from PaddockTS.IndicesAndVegFrac.veg_frac import compute_fractional_cover
                    ds_vegfrac = compute_fractional_cover(query, ds_sentinel2=ds_sentinel2)
                elif i == 3:
                    from PaddockTS.Plotting.sentinel2_video import sentinel2_video
                    sentinel2_video(query, ds_sentinel2=ds_sentinel2)
                elif i == 4:
                    import geopandas as gpd
                    gpkg_path = f'{query.tmp_dir}/{query.stub}_paddocks.gpkg'
                    if exists(gpkg_path):
                        paddocks = gpd.read_file(gpkg_path)
                    else:
                        from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks
                        paddocks = get_paddocks(query, ds_sentinel2=ds_sentinel2)
                elif i == 5:
                    from PaddockTS.Plotting.sentinel2_paddocks_video import sentinel2_video_with_paddocks
                    sentinel2_video_with_paddocks(query, paddocks, ds_sentinel2=ds_sentinel2)
                elif i == 6:
                    from PaddockTS.Plotting.vegfrac_video import vegfrac_video
                    vegfrac_video(query, ds_vegfrac=ds_vegfrac)
                elif i == 7:
                    from PaddockTS.Plotting.vegfrac_paddocks_video import vegfrac_video_with_paddocks
                    vegfrac_video_with_paddocks(query, paddocks, ds_vegfrac=ds_vegfrac, ds_sentinel2=ds_sentinel2)
                elif i == 8:
                    from PaddockTS.PaddockTS.make_paddockTS import make_paddockTS
                    ds_paddockTS = make_paddockTS(query, ds_sentinel2=ds_sentinel2, paddocks=paddocks)
                elif i == 9:
                    from PaddockTS.PaddockTS.make_yearly_paddockTS import make_yearly_paddockTS
                    ds_yearly = make_yearly_paddockTS(query, ds_paddockTS=ds_paddockTS)
                elif i == 10:
                    from PaddockTS.Phenology.estimate_phenology import estimate_phenology
                    phenology_results = estimate_phenology(query, ds_yearly=ds_yearly)
                elif i == 11:
                    from PaddockTS.Plotting.calendar_plot import calendar_plot
                    calendar_plot(query, ds_sentinel2=ds_sentinel2, paddocks=paddocks)
                elif i == 12:
                    from PaddockTS.Plotting.phenology_plot import phenology_plot
                    phenology_plot(query, phenology_results=phenology_results, ds_yearly=ds_yearly, ds_paddockTS=ds_paddockTS)
            statuses[i] = 'done'
        except Exception:
            statuses[i] = 'error'
            times[i] = time.time() - t0
            log.close()
            raise
        times[i] = time.time() - t0
    log.close()


def get_outputs(query: Query, reload: bool = False):
    if reload:
        if exists(query.tmp_dir):
            shutil.rmtree(query.tmp_dir)
        if exists(query.out_dir):
            shutil.rmtree(query.out_dir)

    env_statuses = ['pending'] * len(ENV_STEPS)
    env_times = [None] * len(ENV_STEPS)
    s2_statuses = ['pending'] * len(S2_STEPS)
    s2_times = [None] * len(S2_STEPS)

    errors = []

    def env_worker():
        try:
            _run_env_steps(query, env_statuses, env_times)
        except Exception as e:
            errors.append(('Environmental', e))

    def s2_worker():
        try:
            _run_s2_steps(query, s2_statuses, s2_times)
        except Exception as e:
            errors.append(('Sentinel-2 → PaddockTS', e))

    t_env = threading.Thread(target=env_worker)
    t_s2 = threading.Thread(target=s2_worker)
    t_env.start()
    t_s2.start()

    def render():
        env_table = _make_table('Environmental', ENV_STEPS, env_statuses, env_times)
        s2_table = _make_table('Sentinel-2 → PaddockTS', S2_STEPS, s2_statuses, s2_times)
        return Columns([env_table, s2_table], padding=(0, 4))

    with Live(render(), console=_console, refresh_per_second=4) as live:
        while t_env.is_alive() or t_s2.is_alive():
            live.update(render())
            time.sleep(0.25)
        live.update(render())

    if errors:
        for name, e in errors:
            _console.print(f'[red]{name}: FAILED — {e}[/red]')
        raise errors[0][1]


if __name__ == '__main__':
    import sys
    from PaddockTS.utils import get_example_query
    get_outputs(get_example_query(), reload='--reload' in sys.argv)
