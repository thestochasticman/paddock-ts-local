"""
Stage 2: Watershed segmentation on cluster edges.

Uses the cluster edge map from presegmentation as the elevation map
for watershed, splitting clusters into spatially connected paddocks.
"""

from os.path import exists

import numpy as np
import geopandas as gpd
import rioxarray
import xarray as xr
from shapely.geometry import shape
from rasterio import features
from scipy import ndimage
from skimage import segmentation, morphology

from PaddockTS.query import Query


def run_watershed(
    image: np.ndarray,
    edge_band: int = 1,
    marker_percentile: float = 25,
) -> np.ndarray:
    """
    Run watershed segmentation using cluster edges as barriers.

    Args:
        image: Input image (H, W, C) with cluster labels and edges
        edge_band: Index of the edge magnitude band (default 1)
        marker_percentile: Percentile of lowest edge values for seeds

    Returns:
        Label array with shape (H, W)
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

    h, w = image.shape[:2]

    # Get edge band as elevation map
    if image.ndim == 3 and image.shape[2] > edge_band:
        edges = image[:, :, edge_band]
    else:
        edges = image[:, :, 0]

    edges = np.nan_to_num(edges, nan=1.0)

    # Create markers from low-edge regions
    threshold = np.percentile(edges, marker_percentile)
    markers = edges <= threshold

    # Clean up markers
    markers = morphology.remove_small_objects(markers, min_size=50)
    markers = morphology.remove_small_holes(markers, area_threshold=25)
    markers = morphology.binary_erosion(markers, morphology.disk(3))

    # Label connected marker regions
    markers_labeled, n_markers = ndimage.label(markers)

    if n_markers == 0:
        return np.zeros((h, w), dtype=np.int32)

    print(f"Watershed with {n_markers} seed regions")

    # Run watershed
    labels = segmentation.watershed(edges, markers_labeled)

    return labels


def segments_to_polygons(
    segments: np.ndarray,
    transform,
    crs,
) -> gpd.GeoDataFrame:
    """Convert segment label array to GeoDataFrame of polygons."""
    shapes_gen = features.shapes(
        segments.astype(np.int32),
        transform=transform,
    )

    records = []
    for geom, value in shapes_gen:
        if value == 0:
            continue
        records.append({
            'geometry': shape(geom),
            'segment_id': int(value),
        })

    if not records:
        return gpd.GeoDataFrame(columns=['geometry', 'segment_id'], crs=crs)

    return gpd.GeoDataFrame(records, crs=crs)


def filter_polygons(
    gdf: gpd.GeoDataFrame,
    min_area_ha: float = 5,
    max_area_ha: float = 1500,
    min_compactness: float = 0.1,
) -> gpd.GeoDataFrame:
    """Filter polygons by area and shape metrics."""
    if len(gdf) == 0:
        return gdf

    gdf = gdf.copy()
    gdf['area_ha'] = gdf.geometry.area / 10000
    gdf['perimeter'] = gdf.geometry.length
    gdf['compactness'] = (4 * np.pi * gdf.geometry.area) / (gdf['perimeter'] ** 2)

    mask = (
        (gdf['area_ha'] >= min_area_ha) &
        (gdf['area_ha'] <= max_area_ha) &
        (gdf['compactness'] >= min_compactness)
    )

    gdf_filtered = gdf[mask].reset_index(drop=True)
    gdf_filtered['paddock_id'] = range(1, len(gdf_filtered) + 1)
    gdf_filtered = gdf_filtered.drop(columns=['segment_id'])

    return gdf_filtered


def segment(
    query: Query,
    min_area_ha: float = 5,
    max_area_ha: float = 1500,
    min_compactness: float = 0.1,
    marker_percentile: float = 25,
) -> None:
    """
    Main segmentation pipeline using watershed on cluster edges.

    Args:
        query: Query object with paths
        min_area_ha: Minimum paddock area in hectares
        max_area_ha: Maximum paddock area in hectares
        min_compactness: Minimum shape compactness 0-1
        marker_percentile: Percentile for watershed markers
    """
    path_preseg = query.path_preseg_tif
    path_output = query.path_polygons

    if not exists(path_preseg):
        raise FileNotFoundError(
            f"Presegmentation GeoTIFF not found at {path_preseg}. "
            "Run presegment(query) first."
        )

    # Load preseg image
    data_xr = xr.open_dataarray(path_preseg)
    image = data_xr.values
    if image.ndim == 3 and image.shape[0] <= 4:
        image = np.moveaxis(image, 0, -1)

    transform = data_xr.rio.transform()
    crs = data_xr.rio.crs

    # Calculate total area
    h, w = image.shape[:2]
    pixel_area = abs(transform.a * transform.e)
    total_area_ha = (h * w * pixel_area) / 10000

    print(f"\n=== Watershed Segmentation ===")

    # Run watershed
    segments = run_watershed(image, marker_percentile=marker_percentile)

    # Convert to polygons
    gdf = segments_to_polygons(segments, transform, crs)
    gdf_filtered = filter_polygons(
        gdf,
        min_area_ha=min_area_ha,
        max_area_ha=max_area_ha,
        min_compactness=min_compactness,
    )

    n_paddocks = len(gdf_filtered)
    if n_paddocks > 0:
        total_paddock_area = gdf_filtered['area_ha'].sum()
        coverage = (total_paddock_area / total_area_ha) * 100

        print(f"\n==> Result: {n_paddocks} paddocks, {coverage:.1f}% coverage")
        print(f"    Total paddock area: {total_paddock_area:.1f} ha")
        print(f"    Paddock sizes (ha): min={gdf_filtered['area_ha'].min():.1f}, "
              f"max={gdf_filtered['area_ha'].max():.1f}, "
              f"mean={gdf_filtered['area_ha'].mean():.1f}")
        print("\n    Individual paddocks:")
        for _, row in gdf_filtered.sort_values('area_ha', ascending=False).iterrows():
            print(f"      Paddock {row['paddock_id']:2d}: {row['area_ha']:6.1f} ha  "
                  f"(compactness={row['compactness']:.2f})")

        gdf_filtered.to_file(path_output, driver='GPKG')
    else:
        print("No valid paddocks found")
        gpd.GeoDataFrame(columns=['geometry', 'paddock_id'], crs=crs).to_file(
            path_output, driver='GPKG'
        )

    data_xr.close()


def test():
    from PaddockTS.query import get_example_query

    query = get_example_query()
    segment(query)
    print(f"Output: {query.path_polygons}")
    return exists(query.path_polygons)


if __name__ == '__main__':
    test()
