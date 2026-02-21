import io
import os
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
    'Segment paddocks',
    'Sentinel-2 video',
    'Sentinel-2 + paddocks video',
    'Vegfrac video',
    'Vegfrac + paddocks video',
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
        for path in [query.sentinel2_path, query.vegfrac_path]:
            if exists(path):
                shutil.rmtree(path)
        for suffix in ['_paddocks.gpkg', '_preseg.tif', '_sam_mask.tif', '_sam_raw.gpkg']:
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
        lambda: _compute_vegfrac(query),
        lambda: _segment_paddocks(query),
        lambda: _sentinel2_video(query),
        lambda p=None: _sentinel2_paddocks_video(query, p),
        lambda: _vegfrac_video(query),
        lambda p=None: _vegfrac_paddocks_video(query, p),
    ]

    with Live(Group(_make_table(statuses, times), progress), console=_console, refresh_per_second=4) as live:
        for i, fn in enumerate(step_fns):
            statuses[i] = 'running'
            live.update(Group(_make_table(statuses, times), progress))

            t0 = time.time()
            try:
                if i == 3:
                    live.stop()
                    result = fn()
                    live.start()
                else:
                    with redirect_stdout(open(os.devnull, 'w')):
                        if i == 5:
                            result = _sentinel2_paddocks_video(query, paddocks)
                        elif i == 7:
                            result = _vegfrac_paddocks_video(query, paddocks)
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
    if exists(query.sentinel2_path):
        return
    from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
    download_sentinel2(query)


def _compute_indices(query):
    from PaddockTS.IndicesAndVegFrac.indices import compute_indices
    compute_indices(query)


def _compute_vegfrac(query):
    if exists(query.vegfrac_path):
        return
    from PaddockTS.IndicesAndVegFrac.veg_frac import compute_fractional_cover
    compute_fractional_cover(query)


def _segment_paddocks(query):
    import geopandas as gpd
    gpkg_path = f'{query.tmp_dir}/{query.stub}_paddocks.gpkg'
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


def _vegfrac_video(query):
    from PaddockTS.Plotting.vegfrac_video import vegfrac_video
    return vegfrac_video(query)


def _vegfrac_paddocks_video(query, paddocks):
    from PaddockTS.Plotting.vegfrac_paddocks_video import vegfrac_video_with_paddocks
    return vegfrac_video_with_paddocks(query, paddocks)


if __name__ == '__main__':
    import sys
    from PaddockTS.utils import get_example_query
    run(get_example_query(), reload='--reload' in sys.argv)
