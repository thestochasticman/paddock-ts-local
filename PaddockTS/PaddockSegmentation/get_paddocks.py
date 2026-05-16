"""Extract paddock polygons from a Sentinel-2 stack via Segment Anything.

The pipeline runs in three internal stages:

1. **Presegmentation** — derive a single grayscale image from the
   multi-temporal Sentinel-2 stack using NDWI Fourier features, written
   as a GeoTIFF. This collapses time into a representation that
   emphasises stable field boundaries.
2. **SAM mask generation** — feed the presegmented image to
   `segment-geospatial <https://samgeo.gishub.org/>`_ (default backbone:
   SAM ViT-H) and write a mask GeoTIFF + raw polygons GeoPackage.
3. **Vectorisation and filtering** — clean up the polygons, compute
   per-polygon area (ha) and compactness, drop those outside the
   user-defined size or shape limits, and assign sequential paddock IDs.

SAM weights (``sam_vit_h_4b8939.pth``, ~2.4 GB) are downloaded on first
use to ``{config.tmp_dir}/sam_weights`` and cached.
"""

import gc
import os
import sys
import time
import numpy as np
import geopandas as gpd
from datetime import datetime
from os import makedirs
from os.path import exists
from PaddockTS.query import Query
from PaddockTS.config import config
from PaddockTS.PaddockSegmentation._presegment import presegment
from PaddockTS.PaddockSegmentation.check_if_valid_paddocks_exists import check_if_valid_paddocks_exists


def _log(msg):
    print(msg, file=sys.stderr)


def get_paddocks(
    query: Query,
    ds_sentinel2=None,
    min_area_ha: float = 5,
    max_area_ha: float = 1500,
    min_compactness: float = 0.1,
    device: str | None = None,
) -> gpd.GeoDataFrame:
    """End-to-end paddock segmentation: NDWI preseg → SAM masks → filtered polygons.

    Caches each stage's output under ``query.tmp_dir`` so reruns reuse
    work: ``{stub}_preseg.tif``, ``{stub}_sam_mask.tif``,
    ``{stub}_sam_raw.gpkg``, and the final ``{stub}_sam_paddocks.gpkg``.

    Args:
        query: The :class:`PaddockTS.query.Query`.
        ds_sentinel2: Optional in-memory Sentinel-2 dataset. If ``None``,
            ``query.sentinel2_path`` is opened (or downloaded first).
        min_area_ha: Minimum polygon area in hectares; smaller polygons
            are discarded as noise. Default 5 ha.
        max_area_ha: Maximum polygon area in hectares; larger polygons
            (typically the whole-AOI background mask) are discarded.
            Default 1500 ha.
        min_compactness: Minimum isoperimetric compactness
            ``4πA/L²`` ∈ ``[0, 1]``. ``1`` is a circle, low values are
            sliver-like. Default 0.1 drops elongated edge artefacts.
        device: Torch device for SAM. ``None`` lets samgeo pick (CUDA if
            available, else CPU). Pass ``"cpu"`` to force CPU even with
            a GPU present.

    Returns:
        geopandas.GeoDataFrame: One row per paddock, sorted by
        ``area_ha`` descending, with columns ``geometry``, ``area_ha``,
        ``compactness``, and a 1-based ``paddock`` integer ID. Also
        written to ``{query.tmp_dir}/{query.stub}_sam_paddocks.gpkg``.
    """
    # Cache hit: return previously-segmented paddocks for this (bbox, time)
    if check_if_valid_paddocks_exists(query.sam_paddocks_path):
        _log(f"  Paddocks cache hit at {query.sam_paddocks_path}")
        return gpd.read_file(query.sam_paddocks_path)

    # 1. Presegmentation image
    _log("  Preseg: computing NDWI Fourier features...")
    t0 = time.time()
    preseg_path = presegment(query, ds_sentinel2=ds_sentinel2)
    _log(f"  Preseg: done ({time.time() - t0:.1f}s)")

    # 2. SAMGeo segmentation
    if not check_if_valid_paddocks_exists(query.sam_raw_path):
        from samgeo import SamGeo

        gc.collect()
        checkpoint_dir = f"{config.tmp_dir}/sam_weights"

        _log("  SAMGeo: loading model...")
        t0 = time.time()
        sam = SamGeo(model_type="vit_h", automatic=True, device=device, checkpoint_dir=checkpoint_dir)
        _log(f"  SAMGeo: model loaded ({time.time() - t0:.1f}s)")

        _log("  SAMGeo: generating masks...")
        t0 = time.time()
        makedirs(os.path.dirname(query.sam_mask_path), exist_ok=True)
        sam.generate(
            preseg_path,
            query.sam_mask_path,
            batch=True,
            foreground=True,
            erosion_kernel=(3, 3),
            mask_multiplier=255,
        )
        _log(f"  SAMGeo: masks generated ({time.time() - t0:.1f}s)")
        # _SUCCESS marker for the SAM mask TIFF
        with open(f'{query.sam_mask_path}._SUCCESS', 'w') as f:
            f.write(datetime.utcnow().isoformat() + 'Z')

        sam.tiff_to_gpkg(query.sam_mask_path, query.sam_raw_path)
        # _SUCCESS marker for the raw polygons GeoPackage
        with open(f'{query.sam_raw_path}._SUCCESS', 'w') as f:
            f.write(datetime.utcnow().isoformat() + 'Z')
        del sam
        gc.collect()
    else:
        _log("  SAMGeo: using cached results")

    # 3. Filter polygons
    _log("  Filtering polygons...")
    gdf = gpd.read_file(query.sam_raw_path)
    # SAM's tiff_to_gpkg sometimes drops CRS even when the source mask.tif
    # carries one; re-attach from the mask (or preseg) if needed so
    # estimate_utm_crs() doesn't blow up downstream.
    if gdf.crs is None:
        import rioxarray
        for source in (query.sam_mask_path, query.preseg_path):
            if exists(source):
                src_crs = rioxarray.open_rasterio(source).rio.crs
                if src_crs is not None:
                    gdf = gdf.set_crs(src_crs)
                    _log(f"  Re-attached CRS {src_crs} from {source}")
                    break
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    metric = gdf.to_crs(gdf.estimate_utm_crs())
    gdf["area_ha"] = metric.geometry.area / 10000
    gdf["compactness"] = (4 * np.pi * metric.geometry.area) / (metric.geometry.length ** 2)
    paddocks = gdf[
        (gdf["area_ha"] >= min_area_ha)
        & (gdf["area_ha"] <= max_area_ha)
        & (gdf["compactness"] >= min_compactness)
    ].copy()
    paddocks = paddocks.sort_values("area_ha", ascending=False).reset_index(drop=True)
    paddocks["paddock"] = range(1, len(paddocks) + 1)

    makedirs(os.path.dirname(query.sam_paddocks_path), exist_ok=True)
    paddocks.to_file(query.sam_paddocks_path, driver="GPKG")
    # Touch ``<gpkg>._SUCCESS`` *after* the gpkg write completes; its presence
    # is what the next call uses as the cache-validity check.
    with open(f'{query.sam_paddocks_path}._SUCCESS', 'w') as f:
        f.write(datetime.utcnow().isoformat() + 'Z')
    _log(f"  {len(paddocks)} paddocks saved to {query.sam_paddocks_path}")

    return paddocks


def test():
    import xarray as xr
    import rioxarray
    import matplotlib.pyplot as plt
    from PaddockTS.utils import get_example_query

    query = get_example_query()
    paddocks = get_paddocks(query, device="cpu")

    ds = xr.open_zarr(query.sentinel2_path, chunks=None, decode_coords="all")
    nir = ds["nbart_nir_1"].transpose("y", "x", "time").values.astype(np.float32)
    red = ds["nbart_red"].transpose("y", "x", "time").values.astype(np.float32)
    nir[nir == 0] = np.nan
    red[red == 0] = np.nan
    ndvi = (nir - red) / (nir + red)
    ndvi[~np.isfinite(ndvi)] = np.nan
    ndvi_median = np.nanmedian(ndvi, axis=2)
    x, y = ds.x.values, ds.y.values
    extent = [x.min(), x.max(), y.min(), y.max()]

    preseg = rioxarray.open_rasterio(f"{query.tmp_dir}/{query.stub}_preseg.tif")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(preseg.transpose("y", "x", "band").values, extent=extent, origin="upper")
    axes[0].set_title("Preseg (NDWI Fourier)")
    axes[0].axis("off")

    axes[1].imshow(ndvi_median, cmap="RdYlGn", extent=extent, origin="upper")
    axes[1].set_title("Median NDVI")
    axes[1].axis("off")

    axes[2].imshow(ndvi_median, cmap="RdYlGn", extent=extent, origin="upper")
    paddocks.boundary.plot(ax=axes[2], color="red", linewidth=1)
    axes[2].set_title(f"{len(paddocks)} Paddocks")
    axes[2].axis("off")

    plt.tight_layout()
    png_path = f"{query.tmp_dir}/{query.stub}_paddocks.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {png_path}")
    plt.show()

    return len(paddocks) > 0


if __name__ == "__main__":
    print(test())
