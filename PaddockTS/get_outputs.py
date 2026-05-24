import matplotlib
matplotlib.use('Agg')

# Pre-import dask in the main thread *before* worker threads start.
# Otherwise the env and S2 workers race to trigger xarray's lazy
# `from dask.base import is_dask_collection`, which can hit a
# partially-initialized `dask.base` module and raise ImportError.
import dask
import dask.base  # noqa: F401
import dask.distributed  # noqa: F401

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
    'Download Sentinel-2',              # 0
    'Compute indices',                  # 1
    'Compute fractional cover',         # 2
    'Sentinel-2 video',                 # 3
    'Segment paddocks (SAM)',           # 4 - skipped if skip_sam
    'S2 + paddocks video (SAM)',        # 5 - skipped if skip_sam
    'S2 + paddocks video (user)',       # 6 - skipped if no paddocks_filepath
    'Fractional cover video',           # 7
    'FC + paddocks video (SAM)',        # 8 - skipped if skip_sam
    'FC + paddocks video (user)',       # 9 - skipped if no paddocks_filepath
    'Make paddock TS (SAM)',            # 10 - skipped if skip_sam
    'Make paddock TS (user)',           # 11 - skipped if no paddocks_filepath
    'Make yearly paddock TS (SAM)',     # 12 - skipped if skip_sam
    'Make yearly paddock TS (user)',    # 13 - skipped if no paddocks_filepath
    'Estimate phenology (SAM)',         # 14 - skipped if skip_sam
    'Estimate phenology (user)',        # 15 - skipped if no paddocks_filepath
    'Calendar plot (SAM)',              # 16 - skipped if skip_sam
    'Calendar plot (user)',             # 17 - skipped if no paddocks_filepath
    'Phenology plot (SAM)',             # 18 - skipped if skip_sam
    'Phenology plot (user)',            # 19 - skipped if no paddocks_filepath
    'Make PDF report',                  # 20
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
        elif status == 'skipped':
            symbol = '[cyan]skipped[/cyan]'
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

def _run_env_steps(query: Query, statuses, times, errors=None):
    os.makedirs(query.tmp_dir, exist_ok=True)
    step_errors = []
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
                if not query.config.email:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Environmental.SILO.download_silo import download_silo
                download_silo(query)
            elif i == 3:
                if not query.config.tern_api_key:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Environmental.SLGASoils.download_slgasoils import download_slga_soils
                download_slga_soils(query)
            elif i == 4:
                from PaddockTS.Plotting.ozwald_plot import ozwald_daily_plot
                ozwald_daily_plot(query)
            elif i == 5:
                if not query.config.email:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Plotting.silo_plot import silo_plot
                silo_plot(query)
            elif i == 6:
                statuses[i] = 'waiting'
                from PaddockTS.Sentinel2.check_if_valid_clean_zarr_exists import check_if_valid_clean_zarr_exists
                while not check_if_valid_clean_zarr_exists(query.sentinel2_clean_path):
                    if errors:
                        raise RuntimeError(
                            'Sentinel-2 worker failed before sentinel2_clean.zarr was produced; '
                            'cannot run terrain plot.'
                        )
                    time.sleep(1)
                statuses[i] = 'running'
                from PaddockTS.Plotting.terrain_tiles_plot import terrain_tiles_plot
                terrain_tiles_plot(query)
            statuses[i] = 'done'
        except Exception as e:
            statuses[i] = 'error'
            times[i] = time.time() - t0
            step_errors.append((ENV_STEPS[i], e))
            continue
        times[i] = time.time() - t0

    # If any steps failed, raise the first error
    if step_errors:
        raise step_errors[0][1]


# Each downstream step is gated on the success of these prior step indices.
# A step is marked 'skipped' (not run) when any of its dependencies has
# status 'error' or 'skipped', preventing one real failure from cascading
# into a wall of red errors against every dependent step.
_S2_STEP_DEPS = {
    # SAM chain — depend on step 4 producing a valid paddocks gpkg
    5:  [4],   # SAM S2+paddocks video
    8:  [4],   # SAM FC+paddocks video
    10: [4],   # SAM per-paddock TS
    16: [4],   # SAM calendar plot
    # SAM TS chain
    12: [10],  # SAM yearly TS
    14: [12],  # SAM phenology
    18: [14],  # SAM phenology plot
    # User TS chain (user gpkg is user-provided, so step 11 has no upstream)
    13: [11],  # user yearly TS
    15: [13],  # user phenology
    19: [15],  # user phenology plot
}


def _run_s2_steps(query, statuses, times, paddocks_filepath=None, skip_sam=False, label_col=None):
    import xarray as xr
    from PaddockTS.Sentinel2.check_if_valid_zarr_exists import check_if_valid_zarr_exists
    from PaddockTS.PaddockSegmentation.check_if_valid_paddocks_exists import check_if_valid_paddocks_exists

    ds_sentinel2 = None
    ds_fractional_cover = None
    gpkg_path = query.sam_paddocks_path

    # SAM-based datasets
    ds_paddockTS = None
    ds_yearly = None
    phenology_results = None

    # User-based datasets
    ds_paddockTS_user = None
    ds_yearly_user = None
    phenology_results_user = None

    os.makedirs(query.tmp_dir, exist_ok=True)
    step_errors = []
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

        # Short-circuit if any upstream step failed or was itself skipped.
        deps = _S2_STEP_DEPS.get(i, [])
        if deps and any(statuses[j] in ('error', 'skipped') for j in deps):
            statuses[i] = 'skipped'
            times[i] = time.time() - t0
            continue

        try:
            if i == 0:
                if not check_if_valid_zarr_exists(query.sentinel2_path):
                    from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
                    ds_sentinel2 = download_sentinel2(query)
                else:
                    ds_sentinel2 = xr.open_zarr(query.sentinel2_path, chunks=None, decode_coords="all")
                from PaddockTS.Sentinel2.clean_sentinel2 import clean_sentinel2
                ds_sentinel2 = clean_sentinel2(query, ds_sentinel2=ds_sentinel2)
            elif i == 1:
                from PaddockTS.SpectralIndices.indices import compute_indices
                ds_sentinel2 = compute_indices(query, ds_sentinel2=ds_sentinel2)
            elif i == 2:
                from PaddockTS.FractionalCover.compute_fractional_cover import compute_fractional_cover
                ds_fractional_cover = compute_fractional_cover(query, ds_sentinel2=ds_sentinel2)
            elif i == 3:
                from PaddockTS.Plotting.sentinel2_video import sentinel2_video
                sentinel2_video(query, ds_sentinel2=ds_sentinel2)

            # Step 4: Segment paddocks (SAM)
            elif i == 4:
                if skip_sam:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                elif not check_if_valid_paddocks_exists(gpkg_path):
                    from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks
                    get_paddocks(query, ds_sentinel2=ds_sentinel2)

            # Step 5: S2 + paddocks video (SAM)
            elif i == 5:
                if skip_sam:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Plotting.sentinel2_paddocks_video import sentinel2_video_with_paddocks
                sentinel2_video_with_paddocks(query, paddocks_filepath=gpkg_path, ds_sentinel2=ds_sentinel2)

            # Step 6: S2 + paddocks video (user)
            elif i == 6:
                if paddocks_filepath is None:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Plotting.sentinel2_paddocks_video import sentinel2_video_with_paddocks
                sentinel2_video_with_paddocks(query, paddocks_filepath=paddocks_filepath, ds_sentinel2=ds_sentinel2, label_col=label_col)

            # Step 7: Fractional cover video
            elif i == 7:
                from PaddockTS.Plotting.fractional_cover_video import fractional_cover_video
                fractional_cover_video(query, ds_fractional_cover=ds_fractional_cover)

            # Step 8: FC + paddocks video (SAM)
            elif i == 8:
                if skip_sam:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Plotting.fractional_cover_paddocks_video import fractional_cover_paddocks_video
                fractional_cover_paddocks_video(query, paddocks_filepath=gpkg_path, ds_fractional_cover=ds_fractional_cover, ds_sentinel2=ds_sentinel2)

            # Step 9: FC + paddocks video (user)
            elif i == 9:
                if paddocks_filepath is None:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Plotting.fractional_cover_paddocks_video import fractional_cover_paddocks_video
                fractional_cover_paddocks_video(query, paddocks_filepath=paddocks_filepath, ds_fractional_cover=ds_fractional_cover, ds_sentinel2=ds_sentinel2, label_col=label_col)

            # Step 10: Make paddock TS (SAM)
            elif i == 10:
                if skip_sam:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Phenology.make_paddock_time_series import make_paddock_time_series
                ds_paddockTS = make_paddock_time_series(query, ds_sentinel2=ds_sentinel2, paddocks_filepath=gpkg_path)

            # Step 11: Make paddock TS (user)
            elif i == 11:
                if paddocks_filepath is None:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Phenology.make_paddock_time_series import make_paddock_time_series
                ds_paddockTS_user = make_paddock_time_series(query, ds_sentinel2=ds_sentinel2, paddocks_filepath=paddocks_filepath)

            # Step 12: Make yearly paddock TS (SAM)
            elif i == 12:
                if skip_sam:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Phenology.make_yearly_paddock_time_series import make_yearly_paddock_time_series
                ds_yearly = make_yearly_paddock_time_series(query, ds_paddockTS=ds_paddockTS, paddocks_filepath=gpkg_path)

            # Step 13: Make yearly paddock TS (user)
            elif i == 13:
                if paddocks_filepath is None:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Phenology.make_yearly_paddock_time_series import make_yearly_paddock_time_series
                ds_yearly_user = make_yearly_paddock_time_series(query, ds_paddockTS=ds_paddockTS_user, paddocks_filepath=paddocks_filepath)

            # Step 14: Estimate phenology (SAM)
            elif i == 14:
                if skip_sam:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Phenology.estimate_phenology import estimate_phenology
                phenology_results = estimate_phenology(query, ds_yearly=ds_yearly)

            # Step 15: Estimate phenology (user)
            elif i == 15:
                if paddocks_filepath is None:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Phenology.estimate_phenology import estimate_phenology
                phenology_results_user = estimate_phenology(query, ds_yearly=ds_yearly_user)

            # Step 16: Calendar plot (SAM)
            elif i == 16:
                if skip_sam:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Plotting.calendar_plot import calendar_plot
                calendar_plot(query, ds_sentinel2=ds_sentinel2, paddocks_filepath=gpkg_path)

            # Step 17: Calendar plot (user)
            elif i == 17:
                if paddocks_filepath is None:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Plotting.calendar_plot import calendar_plot
                calendar_plot(query, ds_sentinel2=ds_sentinel2, paddocks_filepath=paddocks_filepath, label_col=label_col)

            # Step 18: Phenology plot (SAM)
            elif i == 18:
                if skip_sam:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Plotting.phenology_plot import phenology_plot
                phenology_plot(query, phenology_results=phenology_results, ds_yearly=ds_yearly, ds_paddockTS=ds_paddockTS, paddocks_filepath=gpkg_path)

            # Step 19: Phenology plot (user)
            elif i == 19:
                if paddocks_filepath is None:
                    statuses[i] = 'skipped'
                    times[i] = time.time() - t0
                    continue
                from PaddockTS.Plotting.phenology_plot import phenology_plot
                phenology_plot(query, phenology_results=phenology_results_user, ds_yearly=ds_yearly_user, ds_paddockTS=ds_paddockTS_user, paddocks_filepath=paddocks_filepath, label_col=label_col)

            # Step 20: Make PDF report
            elif i == 20:
                from PaddockTS.Plotting.make_pdf import make_pdf
                make_pdf(query, paddocks_filepath=paddocks_filepath,
                         label_col=label_col)

            statuses[i] = 'done'
        except Exception as e:
            statuses[i] = 'error'
            times[i] = time.time() - t0
            step_errors.append((S2_STEPS[i], e))
            continue
        times[i] = time.time() - t0

    # If any steps failed, raise the first error
    if step_errors:
        raise step_errors[0][1]


# --- driver ----------------------------------------------------------------

def get_outputs(query: Query, reload: bool = False, show_log: bool = False,
                paddocks_filepath: str = None, skip_sam: bool = False,
                label_col: str | None = None):
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
    - **Sentinel-2 → PaddockTS** (21 steps): S2 download + clean,
      indices, fractional cover, S2 video, paddock segmentation,
      paddock-overlay videos (SAM and user variants), paddockTS
      aggregation, yearly split, phenology, calendar and phenology
      plots (SAM and user variants), and a final PDF report.

    Args:
        query: The :class:`PaddockTS.query.Query` to run the pipeline for.
        reload: If ``True``, delete this query's cached artifacts under
            ``query.query_dir`` (Sentinel-2, indices, fractional cover,
            SAM paddocks), the terrain DEM at ``query.terrain_path``,
            and the per-stub ``query.tmp_dir`` / ``query.out_dir``
            directories before starting. Forces every step to re-run.
            Default ``False``.
        show_log: If ``True``, render a tail-of-log panel below the
            status tables. Useful for debugging stuck steps; turns off
            by default to keep the dashboard compact.
        paddocks_filepath: Optional path to a user-provided paddocks file
            (GeoPackage, Shapefile, or GeoJSON). When given, the user
            variants of the per-paddock TS, plots, and videos are
            produced alongside (or instead of, with ``skip_sam=True``)
            the SAM-based ones.
        skip_sam: If ``True``, skip SAM auto-segmentation. Requires
            ``paddocks_filepath`` to be provided.
        label_col: Column name in the user paddocks file to use for
            human-readable labels in plots and video overlays. If
            ``None``, falls back to the numeric ``paddock`` ID column.

    Raises:
        Exception: Re-raises the first exception encountered by either
            worker (after letting the other finish or fail). Specific
            failures show up in the dashboard as a red ``error`` row
            against the offending step.

    Example:
        ```python
        from datetime import date
        from PaddockTS.query import Query
        from PaddockTS.get_outputs import get_outputs

        q = Query(
            bbox=[148.46, -34.39, 148.50, -34.36],
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
            stub='milgadara',
        )
        get_outputs(q)
        ```
    """
    if skip_sam and paddocks_filepath is None:
        raise ValueError("skip_sam=True requires a valid paddocks_filepath")

    if reload:
        # Per-(bbox, time) caches: S2 raw + clean, indices, fractional cover,
        # presegmentation tif, SAM masks, and the SAM paddocks gpkg. These
        # all live under query_dir (= {tmp_dir}/aoi/{bbox_hash}/{time_hash})
        # which is shared only by Queries with identical (bbox, start, end).
        if exists(query.query_dir):
            shutil.rmtree(query.query_dir)
        # Terrain DEM is per-bbox, time-invariant. Remove the file + marker
        # but leave aoi_dir intact so unrelated time-range queries with the
        # same bbox aren't surprised.
        for path in (query.terrain_path, f'{query.terrain_path}._SUCCESS'):
            if exists(path):
                os.remove(path)
        # Per-stub time-series zarrs + final outputs.
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
                _run_s2_steps(query, s2_statuses, s2_times,
                              paddocks_filepath=paddocks_filepath, skip_sam=skip_sam,
                              label_col=label_col)
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
            screen=show_log,
        ) as live:
            while t_env.is_alive() or t_s2.is_alive():
                live.update(view())
                time.sleep(0.1)
            live.update(view())

        # Live with screen=True uses the alternate screen buffer, which gets
        # torn down on exit and wipes the dashboard. Re-emit the final view
        # to the real terminal so the result persists in scrollback.
        if show_log:
            console.print(view())

    if errors:
        for label, e in errors:
            print(f'{label}: FAILED — {e}')
        raise errors[0][1]
    


if __name__ == '__main__':
    from datetime import date
    fp = 'artifacts/PaddockSet1.gpkg'
    query = Query.build_from_paddocks(fp, date(2024, 1, 1), date(2025, 1, 1), 'PaddockSet1')
    get_outputs(query, reload='--reload' in sys.argv, paddocks_filepath=fp, label_col='paddock', show_log=True)