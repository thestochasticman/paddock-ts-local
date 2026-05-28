"""Per-year calendar of true-colour Sentinel-2 thumbnails per paddock.

Produces one page per year (split across multiple pages if there are
more paddocks than ``max_paddocks_per_page``). Rows are paddocks
(largest area at top); columns are 48 evenly-spaced slots across the
year (4 per month). Each cell shows the Sentinel-2 RGB thumbnail of
that paddock at the observation closest to the slot's day-of-year,
with non-paddock pixels masked black.

Each page is a matplotlib :class:`~matplotlib.figure.Figure` so that
when it's written into a PDF report by :mod:`PaddockTS.Plotting.make_pdf`,
the title / month / paddock labels remain *vector text* — readable at
any zoom, immune to the rasterized-PNG-embed shrink that capped text
size at ~13 pt in the previous PIL-composited version.

Public entry points:

- :func:`calendar_plot` — saves one PNG per page (matplotlib raster
  output) under ``query.out_dir``. Standalone view, same as before.
- :func:`iter_calendar_figures` — generator yielding ``(year, page_idx,
  fig)`` tuples without writing anything to disk. Used by
  :mod:`PaddockTS.Plotting.make_pdf` to embed pages as vector-text PDF
  pages.
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Iterator

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from rasterio.features import rasterize

from PaddockTS.query import Query


# --- thumbnail prep --------------------------------------------------------

def _to_rgb(ds, time_idx):
    r = ds['nbart_red'].isel(time=time_idx).values.astype(np.float32)
    g = ds['nbart_green'].isel(time=time_idx).values.astype(np.float32)
    b = ds['nbart_blue'].isel(time=time_idx).values.astype(np.float32)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[rgb == 0] = np.nan
    rgb /= 10000.0
    rgb = np.clip(rgb * 3, 0, 1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    return rgb


def extract_paddock_thumbnails(
    query: Query,
    paddock_id,
    year: int,
    ds_sentinel2: 'xr.Dataset | None' = None,
    paddocks_filepath: str | None = None,
    thumb_size: int = 256,
    n_slots: int = 48,
) -> dict:
    """Return per-slot RGB thumbnails for one paddock × one year.

    Same per-slot semantics as :func:`calendar_plot` (n_slots evenly-spaced
    day-of-year slots, each filled with the closest observation), but
    extracted just for the requested paddock so it can be served as a
    lightweight web payload rather than a full composite PNG.

    Args:
        query: The :class:`PaddockTS.query.Query`.
        paddock_id: Paddock identifier matching the ``paddock`` column of
            ``paddocks_filepath`` (compared as strings).
        year: Calendar year to extract.
        ds_sentinel2: Optional in-memory cleaned Sentinel-2 dataset. If
            ``None``, ``query.sentinel2_clean_path`` is opened (and produced
            via :func:`clean_sentinel2` if missing).
        paddocks_filepath: Path to the paddocks file. Defaults to
            ``query.sam_paddocks_path``.
        thumb_size: Edge length of each thumbnail in pixels. Default 256.
        n_slots: Number of slots across the year. Default 48.

    Returns:
        dict with keys ``paddock_id`` (str), ``label`` (str), ``area_ha`` (float|None),
        ``year`` (int), ``thumb_size`` (int), ``n_slots`` (int),
        ``dates`` (list[str|None], iso ``YYYY-MM-DD`` per slot),
        ``thumbnails`` (list[np.ndarray], each uint8 ``(H, W, 3)``).
    """
    from PIL import Image
    import rioxarray  # noqa: F401
    from PaddockTS.utils import load_user_paddocks

    if paddocks_filepath is None:
        paddocks_filepath = query.sam_paddocks_path

    if ds_sentinel2 is None:
        from PaddockTS.Sentinel2.check_if_valid_clean_zarr_exists import check_if_valid_clean_zarr_exists
        if not check_if_valid_clean_zarr_exists(query.sentinel2_clean_path):
            from PaddockTS.Sentinel2.clean_sentinel2 import clean_sentinel2
            clean_sentinel2(query)
        ds_sentinel2 = xr.open_zarr(query.sentinel2_clean_path, chunks=None, decode_coords='all')

    paddocks = load_user_paddocks(paddocks_filepath)
    ds_crs = ds_sentinel2.rio.crs
    if paddocks.crs != ds_crs:
        paddocks = paddocks.to_crs(ds_crs)

    paddock_str = str(paddock_id)
    matches = paddocks[paddocks['paddock'].astype(str) == paddock_str]
    if matches.empty:
        raise ValueError(f'paddock_id {paddock_id!r} not found in {paddocks_filepath}')
    row = matches.iloc[0]
    area_ha = float(row['area_ha']) if 'area_ha' in row.index and row['area_ha'] is not None else None
    label = paddock_str

    transform = ds_sentinel2.rio.transform()
    h, w = ds_sentinel2.sizes['y'], ds_sentinel2.sizes['x']
    pid = 1
    mask = rasterize([(row.geometry, pid)], out_shape=(h, w), transform=transform, fill=0, dtype=np.int32)
    ys, xs = np.where(mask == pid)

    black = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
    base = {
        'paddock_id': paddock_str,
        'label': label,
        'area_ha': area_ha,
        'year': int(year),
        'thumb_size': int(thumb_size),
        'n_slots': int(n_slots),
    }
    if len(ys) == 0:
        return {**base, 'dates': [None] * n_slots, 'thumbnails': [black] * n_slots}

    pad = 2
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, h)
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, w)
    mask_crop = mask[y0:y1, x0:x1]

    year_mask = ds_sentinel2.time.dt.year.values == year
    ds_year = ds_sentinel2.isel(time=year_mask)
    if ds_year.sizes['time'] == 0:
        return {**base, 'dates': [None] * n_slots, 'thumbnails': [black] * n_slots}

    obs_doy = ds_year.time.dt.dayofyear.values
    slot_centres = np.linspace(1, 365, n_slots + 1)
    slot_centres = (slot_centres[:-1] + slot_centres[1:]) / 2
    slot_to_obs = [int(np.argmin(np.abs(obs_doy - sc))) for sc in slot_centres]

    obs_times = ds_year.time.values  # numpy datetime64

    obs_thumbs = {}
    for obs_idx in set(slot_to_obs):
        rgb = _to_rgb(ds_year, obs_idx)
        crop = rgb[y0:y1, x0:x1].copy()
        crop[mask_crop != pid] = 0.0
        img = Image.fromarray((crop * 255).astype(np.uint8))
        obs_thumbs[obs_idx] = np.array(img.resize((thumb_size, thumb_size), Image.NEAREST))

    thumbnails = [obs_thumbs[idx] for idx in slot_to_obs]
    dates = [str(obs_times[idx])[:10] for idx in slot_to_obs]
    return {**base, 'dates': dates, 'thumbnails': thumbnails}


def _resolve_ds(query: Query, ds_sentinel2):
    if ds_sentinel2 is not None:
        return ds_sentinel2
    from PaddockTS.Sentinel2.check_if_valid_clean_zarr_exists import check_if_valid_clean_zarr_exists
    if not check_if_valid_clean_zarr_exists(query.sentinel2_clean_path):
        from PaddockTS.Sentinel2.clean_sentinel2 import clean_sentinel2
        clean_sentinel2(query)
    return xr.open_zarr(query.sentinel2_clean_path, chunks=None, decode_coords='all')


def _prepare_thumbnails(query: Query, paddocks_filepath: str,
                        ds_sentinel2, thumb_size: int):
    """Compute the per-paddock thumbnails once, reused across all pages.

    Returns
    -------
    paddocks_sorted : GeoDataFrame
        Paddocks sorted largest-area first.
    paddock_ids : list[int]
        Paddock IDs in the same order as ``paddocks_sorted``.
    years_data : dict[int, tuple[dict, list[int]]]
        ``{year: (obs_thumbs, slot_to_obs)}``. ``obs_thumbs`` is
        ``{obs_idx: {paddock_id: (thumb_size, thumb_size, 3) uint8 array}}``;
        ``slot_to_obs`` is the per-slot index into the year's observations.
    """
    import rioxarray  # noqa: F401
    from PIL import Image
    from PaddockTS.utils import load_user_paddocks

    paddocks = load_user_paddocks(paddocks_filepath)
    ds_crs = ds_sentinel2.rio.crs
    if paddocks.crs != ds_crs:
        paddocks = paddocks.to_crs(ds_crs)

    paddocks_sorted = paddocks.sort_values('area_ha', ascending=False).reset_index(drop=True)
    paddock_ids = [int(row['paddock']) for _, row in paddocks_sorted.iterrows()]

    transform = ds_sentinel2.rio.transform()
    h, w = ds_sentinel2.sizes['y'], ds_sentinel2.sizes['x']
    shapes = [(geom, pid) for geom, pid in zip(paddocks_sorted.geometry, paddock_ids)]
    mask = rasterize(shapes, out_shape=(h, w), transform=transform, fill=0, dtype=np.int32)

    pad = 2
    bboxes = {}
    mask_crops = {}
    for pid in paddock_ids:
        ys, xs = np.where(mask == pid)
        if len(ys) == 0:
            bboxes[pid] = None
        else:
            y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
            x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
            bboxes[pid] = (y0, y1, x0, x1)
            mask_crops[pid] = mask[y0:y1, x0:x1]

    black_thumb = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)

    def _crop_all(rgb):
        thumbs = {}
        for pid in paddock_ids:
            bbox = bboxes[pid]
            if bbox is None:
                thumbs[pid] = black_thumb
                continue
            y0, y1, x0, x1 = bbox
            crop = rgb[y0:y1, x0:x1].copy()
            crop[mask_crops[pid] != pid] = 0.0
            img = Image.fromarray((crop * 255).astype(np.uint8))
            thumbs[pid] = np.array(img.resize((thumb_size, thumb_size), Image.NEAREST))
        return thumbs

    n_slots = 48
    slot_centres = np.linspace(1, 365, n_slots + 1)
    slot_centres = (slot_centres[:-1] + slot_centres[1:]) / 2

    years = np.unique(ds_sentinel2.time.dt.year.values)
    years_data: dict[int, tuple[dict, list[int]]] = {}
    for year in years:
        year_mask = ds_sentinel2.time.dt.year.values == int(year)
        ds_year = ds_sentinel2.isel(time=year_mask)
        if ds_year.sizes['time'] == 0:
            continue
        obs_doy = ds_year.time.dt.dayofyear.values
        slot_to_obs = [int(np.argmin(np.abs(obs_doy - sc))) for sc in slot_centres]
        obs_thumbs = {}
        for obs_idx in set(slot_to_obs):
            rgb = _to_rgb(ds_year, obs_idx)
            obs_thumbs[obs_idx] = _crop_all(rgb)
        years_data[int(year)] = (obs_thumbs, slot_to_obs)

    return paddocks_sorted, paddock_ids, years_data


# --- per-page figure builder ----------------------------------------------

_MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Figure is sized to match the make_pdf landscape-A4 embed area, so PDF
# scaling is ~1:1 and matplotlib font sizes map straight to PDF points.
_FIG_W_IN = 10.89   # matches make_pdf's max_w
_FIG_H_IN = 7.47    # matches make_pdf's max_h
# Generous side margins so the grid sits visually centred on the page
# instead of bleeding to the edges.
_LEFT_MARGIN = 0.18      # fraction of fig width reserved for paddock labels
_RIGHT_MARGIN = 0.08     # blank gutter on the right of the grid
_TITLE_BAND = 0.06       # fraction of fig height for the title strip (top)
_HEADER_BAND = 0.04      # fraction of fig height for the month-name strip
_BOTTOM_MARGIN = 0.04    # blank gutter under the grid


def _build_page_figure(stub: str, year: int, page_idx: int, n_pages: int,
                       page_paddock_ids: list[int], paddocks_sorted,
                       paddock_ids_all: list[int],
                       obs_thumbs: dict, slot_to_obs: list[int],
                       n_slots: int = 48, thumb_size: int = 64,
                       label_col: str | None = None,
                       max_paddocks_per_page: int = 20):
    """Build the matplotlib Figure for one calendar page.

    The thumbnail grid is composited into a single numpy array and drawn
    via one ``imshow`` (fast). Title, month names, and paddock labels
    are matplotlib text — vector when saved to PDF.

    Row height is fixed by ``max_paddocks_per_page`` so a partial last
    page has the same per-row height as the full pages above it.
    """
    n_rows_visible = len(page_paddock_ids)
    grid_h = n_rows_visible * thumb_size
    grid_w = n_slots * thumb_size
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for i, pid in enumerate(page_paddock_ids):
        y0 = i * thumb_size
        for j in range(n_slots):
            x0 = j * thumb_size
            grid[y0:y0 + thumb_size, x0:x0 + thumb_size] = obs_thumbs[slot_to_obs[j]][pid]

    fig = plt.figure(figsize=(_FIG_W_IN, _FIG_H_IN))

    # Horizontal extent of the thumbnail grid (left edge = paddock-label
    # column ends; right edge = right gutter starts).
    grid_left = _LEFT_MARGIN
    grid_right = 1.0 - _RIGHT_MARGIN

    # Vertical: the *available* band sits between the month-name strip
    # (just under the title) and the bottom margin. Per-row height is
    # fixed by max_paddocks_per_page so a half-full last page looks the
    # same as full pages — but for partial pages, the grid is then
    # vertically *centred* in the leftover space so it doesn't dangle at
    # the top.
    band_top    = 1.0 - _TITLE_BAND - _HEADER_BAND
    band_bottom = _BOTTOM_MARGIN
    band_height = band_top - band_bottom
    row_h_frac  = band_height / max_paddocks_per_page
    visible_grid_h = row_h_frac * n_rows_visible
    # Centre the visible grid vertically within the band.
    grid_top    = band_top - (band_height - visible_grid_h) / 2
    grid_bottom = grid_top - visible_grid_h

    grid_ax = fig.add_axes([grid_left, grid_bottom,
                            grid_right - grid_left, visible_grid_h])
    grid_ax.imshow(grid, aspect='auto', interpolation='nearest')
    grid_ax.set_xticks([])
    grid_ax.set_yticks([])
    for spine in grid_ax.spines.values():
        spine.set_visible(False)

    # Title
    title_text = f'{stub} — {year} (page {page_idx + 1:02d}/{n_pages:02d})'
    fig.text(0.5, 1.0 - _TITLE_BAND / 2, title_text,
             ha='center', va='center', fontsize=16, fontweight='bold')

    # Month labels — sit just above the grid (not at a fixed band height)
    # so they follow the grid when it's vertically centred for partial pages.
    header_y = grid_top + _HEADER_BAND / 2
    grid_w_frac = grid_right - grid_left
    for m in range(12):
        slot_left = m * 4 + 0.5   # centre of the leftmost slot of this month
        x = grid_left + (slot_left / n_slots) * grid_w_frac
        fig.text(x, header_y, _MONTH_NAMES[m],
                 ha='center', va='center', fontsize=11)

    # Paddock labels (one per row, right-aligned just to the left of the grid)
    label_x = grid_left - 0.005
    for i, pid in enumerate(page_paddock_ids):
        orig_idx = paddock_ids_all.index(pid)
        row = paddocks_sorted.iloc[orig_idx]
        if label_col is not None:
            label_text = str(row[label_col])
        else:
            label_text = f'P{row["paddock"]}  {row["area_ha"]:.0f}ha'
        y = grid_top - (i + 0.5) * row_h_frac
        fig.text(label_x, y, label_text,
                 ha='right', va='center', fontsize=10)

    return fig


# --- public API ------------------------------------------------------------

def iter_calendar_figures(query: Query, paddocks_filepath: str | None = None,
                          ds_sentinel2: xr.Dataset | None = None,
                          thumb_size: int = 64,
                          max_paddocks_per_page: int = 20,
                          label_col: str | None = None,
                          ) -> Iterator[tuple[int, int, plt.Figure]]:
    """Yield ``(year, page_idx, fig)`` for every calendar page.

    Does not write to disk. Used by :mod:`PaddockTS.Plotting.make_pdf`
    to embed each page directly into the report PDF as a vector-text
    page. The caller is responsible for ``plt.close(fig)`` after
    consuming each Figure.
    """
    if paddocks_filepath is None:
        paddocks_filepath = query.sam_paddocks_path

    ds_sentinel2 = _resolve_ds(query, ds_sentinel2)
    paddocks_sorted, paddock_ids, years_data = _prepare_thumbnails(
        query, paddocks_filepath, ds_sentinel2, thumb_size,
    )

    n_paddocks = len(paddock_ids)
    n_pages = (n_paddocks + max_paddocks_per_page - 1) // max_paddocks_per_page
    paddock_pages = [paddock_ids[i * max_paddocks_per_page:(i + 1) * max_paddocks_per_page]
                     for i in range(n_pages)]

    for year, (obs_thumbs, slot_to_obs) in years_data.items():
        for page_idx, page_paddock_ids in enumerate(paddock_pages):
            fig = _build_page_figure(
                stub=query.stub, year=year, page_idx=page_idx, n_pages=n_pages,
                page_paddock_ids=page_paddock_ids,
                paddocks_sorted=paddocks_sorted, paddock_ids_all=paddock_ids,
                obs_thumbs=obs_thumbs, slot_to_obs=slot_to_obs,
                thumb_size=thumb_size, label_col=label_col,
                max_paddocks_per_page=max_paddocks_per_page,
            )
            yield year, page_idx, fig


def calendar_plot(query: Query, ds_sentinel2: xr.Dataset | None = None,
                  paddocks_filepath: str | None = None,
                  thumb_size: int = 64,
                  max_paddocks_per_page: int = 20,
                  label_col: str | None = None) -> list[str]:
    """Render and save one calendar PNG per year × page chunk.

    The PNGs are matplotlib-rasterized at 200 dpi for standalone
    viewing. For the PDF report, :mod:`PaddockTS.Plotting.make_pdf`
    calls :func:`iter_calendar_figures` directly so the text stays
    vector.

    Args:
        query: The :class:`PaddockTS.query.Query`.
        ds_sentinel2: Optional in-memory cleaned Sentinel-2 dataset. If
            ``None``, opened (or downloaded + cleaned) from
            ``query.sentinel2_clean_path``.
        paddocks_filepath: Path to the paddocks file. If ``None``,
            defaults to ``query.sam_paddocks_path``.
        thumb_size: Edge length of each thumbnail in pixels (input
            resolution; matplotlib resizes for display). Default 64.
        max_paddocks_per_page: Maximum paddocks per page. Default 20.
        label_col: Column in the paddocks GeoDataFrame to use for
            per-row labels. ``None`` → ``"P{id}  {area:.0f}ha"``.

    Returns:
        list[str]: Paths of the generated PNGs (one per year × page).
    """
    if paddocks_filepath is None:
        paddocks_filepath = query.sam_paddocks_path

    out_stem = Path(paddocks_filepath).stem
    os.makedirs(query.out_dir, exist_ok=True)

    # Clean up any existing calendar PNGs for this stem first.
    for old in glob.glob(f'{query.out_dir}/{out_stem}_calendar_*.png'):
        os.remove(old)

    out_paths: list[str] = []
    for year, page_idx, fig in iter_calendar_figures(
        query, paddocks_filepath=paddocks_filepath, ds_sentinel2=ds_sentinel2,
        thumb_size=thumb_size, max_paddocks_per_page=max_paddocks_per_page,
        label_col=label_col,
    ):
        out_path = f'{query.out_dir}/{out_stem}_calendar_{year}_p{page_idx + 1:02d}.png'
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f'Saved to {out_path}')
        out_paths.append(out_path)
    return out_paths


def test():
    from PaddockTS.utils import get_example_query
    query = get_example_query()
    calendar_plot(query)


if __name__ == '__main__':
    test()
