import os
import sys
os.environ["PROJ_DATA"] = os.path.join(sys.prefix, "share", "proj")

import gc
import io
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
    'Segment paddocks',
    'Sentinel-2 video',
    'Sentinel-2 + paddocks video',
    'Fractional cover video',
    'Fractional cover + paddocks video',
]


def _make_table(statuses, times):
    table = Table(title='Sentinel-2 to Paddock Pipeline')
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
    if reload:
        import shutil
        for path in [query.sentinel2_path, query.sentinel2_clean_path, query.fractional_cover_path]:
            if exists(path):
                shutil.rmtree(path)
        for suffix in ['_sam_paddocks.gpkg', '_preseg.tif', '_sam_mask.tif', '_sam_raw.gpkg']:
            path = f'{query.tmp_dir}/{query.stub}{suffix}'
            if exists(path):
                os.remove(path)

    statuses = ['pending'] * len(STEPS)
    times = [None] * len(STEPS)
    paddocks = None

    progress = Progress(
        TextColumn('[bold blue]{task.description}'),
        BarColumn(),
        TextColumn('{task.completed}/{task.total}'),
        TimeElapsedColumn(),
        console=_console,
    )
    task_id = progress.add_task('Pipeline', total=len(STEPS))

    step_fns = [
        lambda: _download_sentinel2(query),
        lambda: _compute_indices(query),
        lambda: _compute_fractional_cover(query),
        lambda: _segment_paddocks(query),
        lambda: _sentinel2_video(query),
        lambda p=None: _sentinel2_paddocks_video(query, p),
        lambda: _fractional_cover_video(query),
        lambda p=None: _fractional_cover_paddocks_video(query, p),
    ]

    with Live(Group(_make_table(statuses, times), progress), console=_console, refresh_per_second=4) as live:
        for i, fn in enumerate(step_fns):
            statuses[i] = 'running'
            live.update(Group(_make_table(statuses, times), progress))

            # Clean up before segmentation step
            if i == 3:
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
                with open(os.devnull, 'w') as _null, redirect_stdout(_null), redirect_stderr(_null):
                    if i == 5:
                        result = _sentinel2_paddocks_video(query, paddocks)
                    elif i == 7:
                        result = _fractional_cover_paddocks_video(query, paddocks)
                    else:
                        result = fn()
                if i == 3:
                    paddocks = result
                statuses[i] = 'done'
            except Exception as e:
                statuses[i] = f'[red]error[/red]'
                times[i] = time.time() - t0
                live.update(Group(_make_table(statuses, times), progress))
                raise

            times[i] = time.time() - t0
            progress.update(task_id, completed=i + 1)
            live.update(Group(_make_table(statuses, times), progress))


def _download_sentinel2(query):
    if not exists(query.sentinel2_path):
        from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
        download_sentinel2(query)
    from PaddockTS.Sentinel2.check_if_valid_clean_zarr_exists import check_if_valid_clean_zarr_exists
    if not check_if_valid_clean_zarr_exists(query.sentinel2_clean_path):
        from PaddockTS.Sentinel2.clean_sentinel2 import clean_sentinel2
        clean_sentinel2(query)


def _compute_indices(query):
    from PaddockTS.SpectralIndices.indices import compute_indices
    compute_indices(query)


def _compute_fractional_cover(query):
    if exists(query.fractional_cover_path):
        return
    from PaddockTS.FractionalCover.compute_fractional_cover import compute_fractional_cover
    compute_fractional_cover(query)


def _segment_paddocks(query):
    import geopandas as gpd
    gpkg_path = f'{query.tmp_dir}/{query.stub}_sam_paddocks.gpkg'
    if exists(gpkg_path):
        return gpd.read_file(gpkg_path)
    from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks
    return get_paddocks(query)


def _sentinel2_video(query):
    from PaddockTS.Plotting.sentinel2_video import sentinel2_video
    return sentinel2_video(query)


def _sentinel2_paddocks_video(query, paddocks):
    from PaddockTS.Plotting.sentinel2_paddocks_video import sentinel2_video_with_paddocks
    return sentinel2_video_with_paddocks(query, paddocks)


def _fractional_cover_video(query):
    from PaddockTS.Plotting.fractional_cover_video import fractional_cover_video
    return fractional_cover_video(query)


def _fractional_cover_paddocks_video(query, paddocks):
    from PaddockTS.Plotting.fractional_cover_paddocks_video import fractional_cover_paddocks_video
    return fractional_cover_paddocks_video(query, paddocks)


if __name__ == '__main__':
    import sys
    from PaddockTS.utils import get_example_query
    run(get_example_query(), reload='--reload' in sys.argv)
