"""Per-year calendar of true-colour Sentinel-2 thumbnails per paddock.

Produces one PNG per year. Rows are paddocks (largest area at top);
columns are 48 evenly-spaced slots across the year (4 per month). Each
cell shows the Sentinel-2 RGB thumbnail of that paddock at the
observation closest to the slot's day-of-year, with non-paddock pixels
masked black. The result is a single composite image — useful for
spotting cloud problems, cropping events, or stand-out paddocks at a
glance.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from rasterio.features import rasterize

from PaddockTS.query import Query


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


def calendar_plot(query: Query, ds_sentinel2: xr.Dataset | None = None, paddocks_filepath: str | None = None, thumb_size: int = 64, max_paddocks_per_page: int = 20, label_col: str | None = None) -> list[str]:
    """Generate one calendar PNG per year of paddock × time-slot thumbnails.

    The image is composited directly as a PIL image (no matplotlib
    axes grid) for memory-efficiency on large paddock counts. Each
    thumbnail is the bbox-cropped, paddock-masked, nearest-neighbour
    resized RGB at the closest observation to the slot centre.

    Args:
        query: The :class:`PaddockTS.query.Query`. Outputs are written
            to ``{query.out_dir}/{paddocks_stem}_calendar_{year}.png``.
        ds_sentinel2: Optional in-memory Sentinel-2 dataset. If ``None``,
            ``query.sentinel2_path`` is opened (or downloaded first).
        paddocks_filepath: Path to the paddocks file. If ``None``, uses
            SAM paddocks from ``{query.tmp_dir}/{query.stub}_sam_paddocks.gpkg``.
        thumb_size: Edge length of each thumbnail in pixels. Default 64.
            Larger values produce sharper but heavier images.
        max_paddocks_per_page: Maximum number of paddocks per output image.
            Default 20. Prevents images from becoming too tall with many
            paddocks.

    Returns:
        list[str]: Filesystem paths of the generated PNGs (one per
        year per page that has at least one observation).
    """
    import os
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(query.out_dir, exist_ok=True)

    from PaddockTS.utils import load_user_paddocks

    # Default to SAM paddocks if no filepath provided
    if paddocks_filepath is None:
        paddocks_filepath = f'{query.tmp_dir}/{query.stub}_sam_paddocks.gpkg'

    out_stem = Path(paddocks_filepath).stem

    if ds_sentinel2 is None:
        if not os.path.exists(query.sentinel2_path):
            from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
            download_sentinel2(query)
        ds_sentinel2 = xr.open_zarr(query.sentinel2_path, chunks=None)

    paddocks = load_user_paddocks(paddocks_filepath)

    # Reproject paddocks to match the dataset CRS
    import rioxarray  # noqa: F401
    ds_crs = ds_sentinel2.rio.crs
    if paddocks.crs != ds_crs:
        paddocks = paddocks.to_crs(ds_crs)

    # Sort paddocks largest to smallest
    paddocks_sorted = paddocks.sort_values('area_ha', ascending=False).reset_index(drop=True)
    n_paddocks = len(paddocks_sorted)
    paddock_ids = [int(row['paddock']) for _, row in paddocks_sorted.iterrows()]

    # Rasterize paddocks to pixel mask
    transform = ds_sentinel2.rio.transform()
    h, w = ds_sentinel2.sizes['y'], ds_sentinel2.sizes['x']
    shapes = [(geom, pid) for geom, pid in zip(paddocks_sorted.geometry, paddock_ids)]
    mask = rasterize(shapes, out_shape=(h, w), transform=transform, fill=0, dtype=np.int32)

    # Precompute bounding boxes and mask crops per paddock (once)
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
        """Crop and resize all paddocks from one RGB frame."""
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

    # 48 slots per year
    n_slots = 48
    slot_centres = np.linspace(1, 365, n_slots + 1)
    slot_centres = (slot_centres[:-1] + slot_centres[1:]) / 2

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Layout constants
    gap = 1
    label_w = thumb_size * 3   # left margin for paddock labels
    header_h = thumb_size       # top margin for month labels
    title_h = thumb_size        # top margin for title

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_title = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
        font_small = font
        font_title = font

    years = np.unique(ds_sentinel2.time.dt.year.values)
    out_paths = []

    # Clean up any existing calendar files for this stem
    import glob
    for old_file in glob.glob(f'{query.out_dir}/{out_stem}_calendar_*.png'):
        os.remove(old_file)

    # Split paddocks into pages
    n_pages = (n_paddocks + max_paddocks_per_page - 1) // max_paddocks_per_page
    paddock_pages = [paddock_ids[i * max_paddocks_per_page:(i + 1) * max_paddocks_per_page] for i in range(n_pages)]

    for year in years:
        year_mask = ds_sentinel2.time.dt.year.values == year
        ds_year = ds_sentinel2.isel(time=year_mask)
        if ds_year.sizes['time'] == 0:
            continue

        obs_doy = ds_year.time.dt.dayofyear.values
        slot_to_obs = [int(np.argmin(np.abs(obs_doy - sc))) for sc in slot_centres]

        # Compute RGB + crop only for unique observations
        obs_thumbs = {}
        for obs_idx in set(slot_to_obs):
            rgb = _to_rgb(ds_year, obs_idx)
            obs_thumbs[obs_idx] = _crop_all(rgb)

        for page_idx, page_paddock_ids in enumerate(paddock_pages):
            n_paddocks_page = len(page_paddock_ids)

            # Composite into one image per page
            canvas_w = label_w + n_slots * (thumb_size + gap)
            canvas_h = title_h + header_h + n_paddocks_page * (thumb_size + gap)
            canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)

            # Title with page number
            title_text = f'{query.stub} — {year} (page {page_idx + 1:02d}/{n_pages:02d})'
            draw.text((canvas_w // 2, title_h // 2), title_text,
                      fill=(0, 0, 0), font=font_title, anchor='mm')

            # Month labels
            for m in range(12):
                x = label_w + m * 4 * (thumb_size + gap)
                draw.text((x, title_h + header_h // 2), month_names[m],
                          fill=(0, 0, 0), font=font, anchor='lm')

            # Paddock labels and thumbnails for this page
            for i, pid in enumerate(page_paddock_ids):
                # Find the original index in paddocks_sorted
                orig_idx = paddock_ids.index(pid)
                row = paddocks_sorted.iloc[orig_idx]
                y_pos = title_h + header_h + i * (thumb_size + gap)

                # Label
                if label_col is not None:
                    label_text = str(row[label_col])
                else:
                    label_text = f'P{row["paddock"]}  {row["area_ha"]:.0f}ha'
                draw.text((label_w - 4, y_pos + thumb_size // 2),
                          label_text, fill=(0, 0, 0), font=font_small, anchor='rm')

                # Thumbnails
                for j in range(n_slots):
                    thumb = obs_thumbs[slot_to_obs[j]][pid]
                    x_pos = label_w + j * (thumb_size + gap)
                    canvas.paste(Image.fromarray(thumb), (x_pos, y_pos))

            # Output filename always includes page number
            out_path = f'{query.out_dir}/{out_stem}_calendar_{year}_p{page_idx + 1:02d}.png'
            canvas.save(out_path)
            print(f'Saved to {out_path}')
            out_paths.append(out_path)

    return out_paths


def test():
    from PaddockTS.utils import get_example_query

    query = get_example_query()
    # calendar_plot(query)
    from datetime import date

    query = Query.build_from_paddocks('/borevitz_projects/data/manual_downloads/Milgadara_paddock-polygons_2024-12-17_12-45-58.json', date(2024, 1, 1), date(2025, 1, 1), 'Milgadara')
    # get_outputs(query, reload='--reload' in sys.argv, paddocks_filepath='/borevitz_projects/data/manual_downloads/Milgadara_paddock-polygons_2024-12-17_12-45-58.json')

    calendar_plot(query, paddocks_filepath='/borevitz_projects/data/manual_downloads/Milgadara_paddock-polygons_2024-12-17_12-45-58.json')
    
    calendar_plot(query)

if __name__ == '__main__':
    test()
