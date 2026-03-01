from __future__ import annotations

import numpy as np
import xarray as xr
import geopandas as gpd
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


def calendar_plot(query: Query, ds_sentinel2: xr.Dataset | None = None, paddocks: gpd.GeoDataFrame | None = None, thumb_size: int = 64) -> list[str]:
    """
    Per-year calendar showing actual sentinel2 RGB thumbnails per paddock.
    Rows: paddocks (largest to smallest). Columns: 48 slots (4 per month).
    One plot per year. Composited directly as a PIL image (no matplotlib axes grid).
    """
    import os
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(query.out_dir, exist_ok=True)

    if ds_sentinel2 is None:
        if not os.path.exists(query.sentinel2_path):
            from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
            download_sentinel2(query)
        ds_sentinel2 = xr.open_zarr(query.sentinel2_path, chunks=None)

    if paddocks is None:
        gpkg_path = f'{query.tmp_dir}/{query.stub}_paddocks.gpkg'
        if os.path.exists(gpkg_path):
            paddocks = gpd.read_file(gpkg_path)
        else:
            from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks
            paddocks = get_paddocks(query, ds_sentinel2=ds_sentinel2)

    # Sort paddocks largest to smallest
    paddocks_sorted = paddocks.sort_values('area_ha', ascending=False).reset_index(drop=True)
    n_paddocks = len(paddocks_sorted)
    paddock_ids = [int(row['paddock']) for _, row in paddocks_sorted.iterrows()]

    # Rasterize paddocks to pixel mask
    import rioxarray  # noqa: F401
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

        # Composite into one large image
        canvas_w = label_w + n_slots * (thumb_size + gap)
        canvas_h = title_h + header_h + n_paddocks * (thumb_size + gap)
        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Title
        draw.text((canvas_w // 2, title_h // 2), f'{query.stub} — {year}',
                  fill=(0, 0, 0), font=font_title, anchor='mm')

        # Month labels
        for m in range(12):
            x = label_w + m * 4 * (thumb_size + gap)
            draw.text((x, title_h + header_h // 2), month_names[m],
                      fill=(0, 0, 0), font=font, anchor='lm')

        # Paddock labels and thumbnails
        for i, pid in enumerate(paddock_ids):
            row = paddocks_sorted.iloc[i]
            y_pos = title_h + header_h + i * (thumb_size + gap)

            # Label
            draw.text((label_w - 4, y_pos + thumb_size // 2),
                      f'P{row["paddock"]}  {row["area_ha"]:.0f}ha',
                      fill=(0, 0, 0), font=font_small, anchor='rm')

            # Thumbnails
            for j in range(n_slots):
                thumb = obs_thumbs[slot_to_obs[j]][pid]
                x_pos = label_w + j * (thumb_size + gap)
                canvas.paste(Image.fromarray(thumb), (x_pos, y_pos))

        out_path = f'{query.out_dir}/{query.stub}_calendar_{year}.png'
        canvas.save(out_path)
        print(f'Saved to {out_path}')
        out_paths.append(out_path)

    return out_paths


def test():
    from PaddockTS.utils import get_example_query

    query = get_example_query()
    calendar_plot(query)


if __name__ == '__main__':
    test()
