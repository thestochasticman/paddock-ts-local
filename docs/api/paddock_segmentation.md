# Paddock segmentation

SAM-based segmentation pipeline that turns a multi-temporal Sentinel-2
stack into a `geopandas.GeoDataFrame` of paddock polygons, with
`area_ha`, `compactness`, and a 1-based `paddock` integer ID.

Three internal stages:

1. **Presegmentation** — derives a single grayscale image from the
   Sentinel-2 stack using NDWI Fourier features. This emphasises
   stable field boundaries and suppresses transient noise (clouds,
   shadows, seasonal greenness). Written as a GeoTIFF at
   `query.preseg_path`.
2. **SAM mask generation** — feeds the presegmented image to
   [`segment-geospatial`](https://samgeo.gishub.org/) (`samgeo`).
   Default backbone is SAM ViT-H (`sam_vit_h_4b8939.pth`, ~2.4 GB)
   which is auto-downloaded on first use to
   `{config.tmp_dir}/sam_weights`. Outputs a mask GeoTIFF and a raw
   polygons GeoPackage.
3. **Vectorisation and filtering** — explodes multipart geometries,
   reprojects to a local UTM zone for accurate area / perimeter,
   computes `area_ha` and isoperimetric `compactness = 4πA/L²`, drops
   polygons outside `[min_area_ha, max_area_ha]` or below
   `min_compactness`, sorts by area descending, and assigns 1-based
   `paddock` IDs. Result written to `query.sam_paddocks_path`.

---

## Example: end-to-end with defaults

```python
from datetime import date
from PaddockTS.query import Query
from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks

q = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2024, 1, 1),
    end=date(2024, 12, 31),
    stub="seg_demo",
)

gdf = get_paddocks(q)         # downloads S2 first if needed
print(gdf[["paddock", "area_ha", "compactness"]].head())

#    paddock  area_ha  compactness
# 0        1   142.30         0.65
# 1        2    87.41         0.71
# 2        3    66.18         0.58
# ...

# Plot over a basemap
gdf.plot(facecolor="none", edgecolor="red", linewidth=1)
```

---

## Example: tuning the filters

Defaults drop polygons under 5 ha, over 1500 ha, or below 0.1 isoperimetric
compactness (sliver-like). For a high-resolution survey of small
horticultural blocks:

```python
gdf = get_paddocks(
    q,
    min_area_ha=0.5,      # keep small blocks
    max_area_ha=200,
    min_compactness=0.2,  # tighter shape filter
    device="cpu",         # force CPU even if CUDA available
)
```

For a broadacre survey where you want only large rectangular fields:

```python
gdf = get_paddocks(
    q,
    min_area_ha=20,
    max_area_ha=5000,
    min_compactness=0.5,
)
```

---

## Example: visual sanity check against NDVI

A useful eyeball test: overlay the polygons on the median NDVI of the
AOI to confirm boundaries follow real field edges.

```python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from PaddockTS.SpectralIndices.indices import compute_ndvi

ds = xr.open_zarr(q.sentinel2_clean_path, decode_coords="all")
ndvi_median = np.nanmedian(compute_ndvi(ds), axis=2)
extent = [ds.x.min(), ds.x.max(), ds.y.min(), ds.y.max()]

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(ndvi_median, cmap="RdYlGn", vmin=-0.1, vmax=0.9,
          extent=extent, origin="upper")
gdf.boundary.plot(ax=ax, color="red", linewidth=1)
ax.set_title(f"{len(gdf)} paddocks")
ax.axis("off")
```

---

## Bring your own paddocks

If you already have field boundaries from QGIS, a cadastral layer, or
a previous run, skip SAM entirely. The downstream stages
(`make_paddock_time_series`, plotting) accept any
GeoPackage / Shapefile / GeoJSON with a `paddock` column. See
[`get_outputs(..., skip_sam=True, paddocks_filepath=...)`](get_outputs.md)
for the orchestrator-level option, or just pass the file directly:

```python
from PaddockTS.Phenology.make_paddock_time_series import make_paddock_time_series

ts = make_paddock_time_series(
    q,
    paddocks_filepath="/path/to/my_paddocks.gpkg",
)
```

`PaddockTS.utils.load_user_paddocks` will add `paddock`, `area_ha`,
and `compactness` columns if your file is missing them.

---

## Reference

::: PaddockTS.PaddockSegmentation.get_paddocks.get_paddocks
