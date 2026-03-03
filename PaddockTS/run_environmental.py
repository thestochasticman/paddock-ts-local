import glob
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
    'Download terrain',
    'Download OzWALD daily',
    'Download SILO',
    'Download SLGA soils',
    'OzWALD plot',
    'SILO plot',
    'Terrain plot',
]


def _make_table(statuses, times):
    table = Table(title='Environmental Pipeline')
    table.add_column('#', style='dim', width=3)
    table.add_column('Step', width=30)
    table.add_column('Status', width=12)
    table.add_column('Time', width=10)
    for i, name in enumerate(STEPS):
        status = statuses[i]
        elapsed = times[i]
        if status == 'done':
            symbol = '[green]done[/green]'
        elif status == 'running':
            symbol = '[yellow]running...[/yellow]'
        elif status == 'waiting':
            symbol = '[cyan]waiting...[/cyan]'
        elif status == 'skipped':
            symbol = '[dim]skipped[/dim]'
        else:
            symbol = '[dim]pending[/dim]'
        time_str = f'{elapsed:.1f}s' if elapsed is not None else ''
        table.add_row(str(i + 1), name, symbol, time_str)
    return table


def run_environmental(query: Query, reload: bool = False, concurrent: bool = False):
    """Download environmental data and produce plots.

    Parameters
    ----------
    reload : bool
        If True, delete cached environmental data and plots before running.
    concurrent : bool
        If True, the terrain plot will wait for sentinel2 data to become
        available (for when running concurrently with sentinel2_to_paddockTS).
    """
    if reload:
        if exists(query.tmp_dir):
            shutil.rmtree(query.tmp_dir)
        if exists(query.out_dir):
            shutil.rmtree(query.out_dir)

    statuses = ['pending'] * len(STEPS)
    times = [None] * len(STEPS)

    progress = Progress(
        TextColumn('[bold blue]{task.description}'),
        BarColumn(),
        TextColumn('{task.completed}/{task.total}'),
        TimeElapsedColumn(),
        console=_console,
    )
    task_id = progress.add_task('Environmental', total=len(STEPS))

    os.makedirs(query.tmp_dir, exist_ok=True)
    log = open(f'{query.tmp_dir}/{query.stub}_environmental.log', 'w')

    with Live(Group(_make_table(statuses, times), progress), console=_console, refresh_per_second=4) as live:
        for i in range(len(STEPS)):
            statuses[i] = 'running'
            live.update(Group(_make_table(statuses, times), progress))

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
                        if concurrent:
                            statuses[i] = 'waiting'
                            live.update(Group(_make_table(statuses, times), progress))
                            while not exists(query.sentinel2_path):
                                time.sleep(1)
                            statuses[i] = 'running'
                            live.update(Group(_make_table(statuses, times), progress))
                        from PaddockTS.Plotting.terrain_tiles_plot import terrain_tiles_plot
                        terrain_tiles_plot(query)

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
    run_environmental(get_example_query(), reload='--reload' in sys.argv)
