import gc
import io
import logging
import os
import shutil
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from os.path import exists

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from PaddockTS.query import Query


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
    'Compute fractional cover',
    'Sentinel-2 video',
    'Segment paddocks',
    'Sentinel-2 + paddocks video',
    'Fractional cover video',
    'Fractional cover + paddocks video',
    'Make paddockTS',
    'Make yearly paddockTS',
    'Estimate phenology',
    'Calendar plot',
    'Phenology plot',
]


# --- log sink + terminal dashboard infra -----------------------------------

class _LogSink(io.TextIOBase):
    """File-like object that buffers writes line-by-line into a shared deque.

    Lines are emitted only when a newline is seen so partial writes don't
    pollute the rendered tail. Thread-safe.
    """

    def __init__(self, buf, stream_name):
        self._buf = buf
        self._stream = stream_name
        self._partial = ''
        self._lock = threading.Lock()

    def write(self, s):
        if not s:
            return 0
        with self._lock:
            self._partial += s
            *lines, self._partial = self._partial.split('\n')
            for line in lines:
                if line:
                    self._buf.append((self._stream, line))
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False

    def writable(self):
        return True


@contextmanager
def _terminal_dashboard():
    """Take exclusive ownership of the terminal for the duration of the block.

    On entry: replaces sys.stdout / sys.stderr with line-buffered ring sinks,
    duplicates fd 2 so rich can write to the real terminal, and pipes the
    original fd 2 into the same ring buffer (catches GDAL/PROJ/TFLite C-level
    chatter). All three are restored on exit.

    Yields (console, log_buf) — Console writes to the terminal directly; the
    deque holds (stream_name, line) tuples for rendering as a log panel.
    """
    log_buf = deque(maxlen=200)

    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    saved_stderr_fd = os.dup(2)        # keep a copy of the real terminal fd

    # Replace Python-level streams
    sys.stdout = _LogSink(log_buf, 'out')
    sys.stderr = _LogSink(log_buf, 'err')

    # Route warnings.warn + Python logging through the same sink
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format='%(levelname)s %(name)s: %(message)s',
        force=True,
    )
    logging.captureWarnings(True)

    # OS-level fd 2 capture: pipe -> drain thread -> ring buffer
    pipe_r, pipe_w = os.pipe()
    os.dup2(pipe_w, 2)
    os.close(pipe_w)

    def _drain_pipe():
        try:
            with os.fdopen(pipe_r, 'r', buffering=1, encoding='utf-8', errors='replace') as f:
                for line in f:
                    log_buf.append(('fd2', line.rstrip('\n')))
        except Exception:
            pass

    drain_thread = threading.Thread(target=_drain_pipe, daemon=True)
    drain_thread.start()

    # Console writes to the saved real-terminal fd
    terminal_file = os.fdopen(saved_stderr_fd, 'w', buffering=1, closefd=False)
    console = Console(file=terminal_file, force_terminal=True)

    try:
        yield console, log_buf
    finally:
        # Restore Python-level streams first
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        # Restore OS fd 2 (pipe writer becomes orphaned, drain thread sees EOF)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        # drain_thread is daemon=True; will exit on EOF or process termination


# --- render helpers --------------------------------------------------------

_STEP_COL_WIDTH = 36
_LOG_PANEL_HEIGHT = 14
_LOG_TAIL_LINES = 12


def _make_table(title, steps, statuses, times):
    table = Table(title=title)
    table.add_column('#', style='dim', width=3)
    table.add_column('Step', width=_STEP_COL_WIDTH)
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


def _make_log_panel(log_buf):
    tail = Text()
    snapshot = list(log_buf)
    for stream, line in snapshot[-_LOG_TAIL_LINES:]:
        if stream == 'err':
            style = 'red'
        elif stream == 'fd2':
            style = 'magenta'
        else:
            style = 'dim'
        tail.append(line + '\n', style=style)
    return Panel(tail, title='log', height=_LOG_PANEL_HEIGHT, border_style='dim')


def _make_view(log_buf, env_statuses, env_times, s2_statuses, s2_times, show_log=False):
    tables = Columns(
        [_make_table('Environmental', ENV_STEPS, env_statuses, env_times),
         _make_table('Sentinel-2 → PaddockTS', S2_STEPS, s2_statuses, s2_times)],
        padding=(0, 4),
    )
    if not show_log:
        return tables
    return Group(tables, _make_log_panel(log_buf))


# --- step runners ----------------------------------------------------------

def _run_env_steps(query, statuses, times, errors=None):
    os.makedirs(query.tmp_dir, exist_ok=True)
    for i in range(len(ENV_STEPS)):
        statuses[i] = 'running'
        t0 = time.time()
        try:
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
                    if errors:
                        raise RuntimeError(
                            'Sentinel-2 worker failed before sentinel2.zarr was produced; '
                            'cannot run terrain plot.'
                        )
                    time.sleep(1)
                statuses[i] = 'running'
                from PaddockTS.Plotting.terrain_tiles_plot import terrain_tiles_plot
                terrain_tiles_plot(query)
            statuses[i] = 'done'
        except Exception:
            statuses[i] = 'error'
            times[i] = time.time() - t0
            raise
        times[i] = time.time() - t0


def _run_s2_steps(query, statuses, times):
    import xarray as xr

    ds_sentinel2 = None
    ds_fractional_cover = None
    paddocks = None
    ds_paddockTS = None
    ds_yearly = None
    phenology_results = None

    os.makedirs(query.tmp_dir, exist_ok=True)
    for i in range(len(S2_STEPS)):
        # Free memory and let torch own all CPU threads before SAM
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
            if i == 0:
                if not exists(query.sentinel2_path):
                    from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
                    ds_sentinel2 = download_sentinel2(query)
                else:
                    ds_sentinel2 = xr.open_zarr(query.sentinel2_path, chunks=None)
            elif i == 1:
                from PaddockTS.SpectralIndices.indices import compute_indices
                ds_sentinel2 = compute_indices(query, ds_sentinel2=ds_sentinel2)
            elif i == 2:
                from PaddockTS.FractionalCover.compute_fractional_cover import compute_fractional_cover
                ds_fractional_cover = compute_fractional_cover(query, ds_sentinel2=ds_sentinel2)
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
                from PaddockTS.Plotting.fractional_cover_video import fractional_cover_video
                fractional_cover_video(query, ds_fractional_cover=ds_fractional_cover)
            elif i == 7:
                from PaddockTS.Plotting.fractional_cover_paddocks_video import fractional_cover_paddocks_video
                fractional_cover_paddocks_video(query, paddocks, ds_fractional_cover=ds_fractional_cover, ds_sentinel2=ds_sentinel2)
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
            raise
        times[i] = time.time() - t0


# --- driver ----------------------------------------------------------------

def get_outputs(query: Query, reload: bool = False, show_log: bool = False):
    """Run the full PaddockTS pipeline for ``query`` with a live status dashboard.

    Spawns two worker threads — one for environmental data
    (terrain / OzWALD / SILO / SLGA) and one for the
    Sentinel-2 → PaddockTS chain — and renders a live two-column
    status table while they run. All stdout, stderr, Python ``logging``,
    ``warnings.warn``, and even C-level fd 2 writes from GDAL/PROJ/TFLite
    are captured into a bounded ring buffer and (optionally) shown in a
    log panel below the tables.

    Each step is cached on disk by the underlying functions, so reruns
    skip work that's already done. Use ``reload=True`` to force a clean
    rebuild.

    Pipeline steps (parallel branches):

    - **Environmental** (7 steps): terrain DEM, OzWALD daily climate,
      SILO climate, SLGA soils, and three diagnostic plots.
    - **Sentinel-2 → PaddockTS** (13 steps): S2 download, indices,
      fractional cover, S2 video, paddock segmentation, paddock-overlay
      videos, paddockTS aggregation, yearly split, phenology, calendar
      and phenology plots.

    Args:
        query: The :class:`PaddockTS.query.Query` to run the pipeline for.
        reload: If ``True``, recursively delete ``query.tmp_dir`` and
            ``query.out_dir`` before starting, forcing every cached step
            to re-run. Default ``False``.
        show_log: If ``True``, render a tail-of-log panel below the
            status tables. Useful for debugging stuck steps; turns off
            by default to keep the dashboard compact.

    Raises:
        Exception: Re-raises the first exception encountered by either
            worker (after letting the other finish or fail). Specific
            failures show up in the dashboard as a red ``error`` row
            against the offending step.

    Example:
        >>> from datetime import date
        >>> from PaddockTS.query import Query
        >>> from PaddockTS.get_outputs import get_outputs
        >>> q = Query(bbox=[148.46, -34.39, 148.50, -34.36],
        ...           start=date(2023, 1, 1), end=date(2023, 12, 31),
        ...           stub='milgadara')
        >>> get_outputs(q)
    """
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

    with _terminal_dashboard() as (console, log_buf):

        def env_worker():
            try:
                _run_env_steps(query, env_statuses, env_times, errors=errors)
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

        view = lambda: _make_view(log_buf, env_statuses, env_times, s2_statuses, s2_times, show_log=show_log)
        with Live(
            view(),
            console=console,
            redirect_stdout=False,
            redirect_stderr=False,
            refresh_per_second=10,
        ) as live:
            while t_env.is_alive() or t_s2.is_alive():
                live.update(view())
                time.sleep(0.1)
            live.update(view())

    if errors:
        for label, e in errors:
            print(f'{label}: FAILED — {e}')
        raise errors[0][1]


if __name__ == '__main__':
    from PaddockTS.utils import get_example_query
    get_outputs(get_example_query(), reload='--reload' in sys.argv)
