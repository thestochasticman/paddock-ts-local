from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
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


def _crop_paddock(rgb, mask, paddock_id, thumb_size=64, pad=2):
    """Crop RGB image to bounding box of a paddock, mask outside to black, resize to uniform square."""
    from PIL import Image

    ys, xs = np.where(mask == paddock_id)
    if len(ys) == 0:
        return np.zeros((thumb_size, thumb_size, 3))
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, rgb.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, rgb.shape[1])
    crop = rgb[y0:y1, x0:x1].copy()
    mask_crop = mask[y0:y1, x0:x1]
    crop[mask_crop != paddock_id] = 0.0
    img = Image.fromarray((crop * 255).astype(np.uint8))
    img = img.resize((thumb_size, thumb_size), Image.NEAREST)
    return np.array(img).astype(np.float32) / 255.0


def calendar_plot(query: Query, ds_sentinel2: xr.Dataset | None = None, paddocks: gpd.GeoDataFrame | None = None) -> list[str]:
    """
    Per-year calendar showing actual sentinel2 RGB thumbnails per paddock.
    Rows: paddocks (largest to smallest). Columns: 48 slots (4 per month).
    One plot per year.
    """
    import os
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

    # Rasterize paddocks to pixel mask
    import rioxarray  # noqa: F401
    transform = ds_sentinel2.rio.transform()
    h, w = ds_sentinel2.sizes['y'], ds_sentinel2.sizes['x']
    shapes = [(geom, int(pid)) for geom, pid in zip(paddocks_sorted.geometry, paddocks_sorted['paddock'])]
    mask = rasterize(shapes, out_shape=(h, w), transform=transform, fill=0, dtype=np.int32)

    # 48 slots per year
    n_slots = 48
    slot_centres = np.linspace(1, 365, n_slots + 1)
    slot_centres = (slot_centres[:-1] + slot_centres[1:]) / 2

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    years = np.unique(ds_sentinel2.time.dt.year.values)
    out_paths = []

    for year in years:
        year_mask = ds_sentinel2.time.dt.year.values == year
        ds_year = ds_sentinel2.isel(time=year_mask)
        n_times = ds_year.sizes['time']
        if n_times == 0:
            continue

        obs_doy = ds_year.time.dt.dayofyear.values

        # Map each slot to nearest observation index
        slot_to_obs = [int(np.argmin(np.abs(obs_doy - sc))) for sc in slot_centres]

        fig, axes = plt.subplots(n_paddocks, n_slots,
                                 figsize=(n_slots * 0.6, n_paddocks * 0.6),
                                 squeeze=False)

        for j in range(n_slots):
            rgb = _to_rgb(ds_year, slot_to_obs[j])
            for i, (_, row) in enumerate(paddocks_sorted.iterrows()):
                ax = axes[i, j]
                thumb = _crop_paddock(rgb, mask, int(row['paddock']))
                ax.imshow(thumb, interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])

                # Paddock label on leftmost column
                if j == 0:
                    ax.set_ylabel(f'P{row["paddock"]}\n{row["area_ha"]:.0f}ha',
                                  fontsize=6, rotation=0, labelpad=30, va='center')

                # Month label on top row
                if i == 0 and j % 4 == 0:
                    ax.set_title(month_names[j // 4], fontsize=7)

        fig.suptitle(f'{query.stub} — {year}', fontsize=10)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        out_path = f'{query.out_dir}/{query.stub}_calendar_{year}.png'
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'Saved to {out_path}')
        out_paths.append(out_path)

    return out_paths


def test():
    from PaddockTS.utils import get_example_query

    query = get_example_query()
    calendar_plot(query)


if __name__ == '__main__':
    test()
