"""
Stage 2: Segmentation using K-means clustering.

This module performs paddock boundary detection using classical
computer vision (no deep learning models required).

Key differences from PaddockSegmentation (SAM-based):
- Uses K-means clustering on spectral features
- No model downloads needed, works offline
- Uses only numpy, scipy, scikit-learn
"""

from os.path import exists
from typing import Optional

import numpy as np
import geopandas as gpd
import rioxarray
import xarray as xr
from shapely.geometry import shape
from rasterio import features
from sklearn.cluster import KMeans
from scipy import ndimage
from skimage import segmentation, morphology
from PaddockTS.query import Query


def run_watershed(
    image: np.ndarray,
    edge_band: int = 2,
    marker_percentile: float = 10,
) -> np.ndarray:
    """
    Run watershed segmentation using edges as barriers.

    This is better for "obvious" paddocks - large uniform areas with
    clear boundaries. Watershed fills regions while respecting edges.

    Args:
        image: Input image (H, W, C) with edge band
        edge_band: Index of the edge magnitude band (default 2)
        marker_percentile: Percentile of lowest edge values to use as seeds.
            E.g., 10 = bottom 10% of edge values become seed regions.

    Returns:
        Label array with shape (H, W)
    """
    # Normalize to [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

    h, w = image.shape[:2]

    # Get edge band as elevation map for watershed
    if image.ndim == 3 and image.shape[2] > edge_band:
        edges = image[:, :, edge_band]
    else:
        # Fallback: compute edges from first band
        edges = np.sqrt(
            ndimage.sobel(image[:, :, 0], axis=0)**2 +
            ndimage.sobel(image[:, :, 0], axis=1)**2
        )
        edges = edges / (edges.max() + 1e-8)

    # Handle NaN
    edges = np.nan_to_num(edges, nan=1.0)  # NaN = high barrier

    # Use percentile-based threshold for robustness
    threshold = np.percentile(edges, marker_percentile)

    # Create markers: seed regions where edges are lowest (most uniform areas)
    markers = edges <= threshold

    # Clean up markers
    markers = morphology.remove_small_objects(markers, min_size=50)
    markers = morphology.remove_small_holes(markers, area_threshold=25)

    # Erode markers to separate touching regions
    markers = morphology.binary_erosion(markers, morphology.disk(2))

    # Label connected marker regions
    markers_labeled, n_markers = ndimage.label(markers)

    if n_markers == 0:
        return np.zeros((h, w), dtype=np.int32)

    # Run watershed - edges are the "elevation" map
    labels = segmentation.watershed(edges, markers_labeled)

    return labels


def find_optimal_k(
    image: np.ndarray,
    transform,
    crs,
    k_range: range = range(3, 12),
    min_area_ha: float = 10,
    max_area_ha: float = 1500,
    min_compactness: float = 0.1,
) -> dict:
    """
    Find optimal number of clusters by maximizing valid paddock coverage.

    This is a task-specific metric: we optimize k based on the quality of
    the final output (filtered polygons), not intermediate clustering metrics.

    For each k, we run the full pipeline and measure:
    - coverage_pct: % of total area covered by valid paddocks
    - n_paddocks: number of valid paddocks detected
    - total_area_ha: total area of valid paddocks

    The optimal k maximizes coverage_pct (the most useful paddock detection).

    Args:
        image: Input image (H, W, C)
        transform: Affine transform for georeferencing
        crs: Coordinate reference system
        k_range: Range of k values to test (default 3-11)
        min_area_ha: Minimum paddock area for filtering
        max_area_ha: Maximum paddock area for filtering
        min_compactness: Minimum compactness for filtering

    Returns:
        dict with 'optimal_k', 'results' (list of metrics per k), and 'total_area_ha'
    """
    # Calculate total image area in hectares
    h, w = image.shape[:2]
    pixel_area = abs(transform.a * transform.e)  # m² per pixel
    total_area_ha = (h * w * pixel_area) / 10000

    results = []

    for k in k_range:
        # Run segmentation
        segments = run_kmeans(image, n_clusters=k)
        gdf = segments_to_polygons(segments, transform, crs)
        gdf_filtered = filter_polygons(
            gdf,
            min_area_ha=min_area_ha,
            max_area_ha=max_area_ha,
            min_compactness=min_compactness,
        )

        # Calculate metrics
        n_paddocks = len(gdf_filtered)
        paddock_area_ha = gdf_filtered['area_ha'].sum() if n_paddocks > 0 else 0
        coverage_pct = (paddock_area_ha / total_area_ha) * 100

        results.append({
            'k': k,
            'n_paddocks': n_paddocks,
            'paddock_area_ha': paddock_area_ha,
            'coverage_pct': coverage_pct,
        })

        print(f"k={k}: {n_paddocks} paddocks, {coverage_pct:.1f}% coverage")

    # Find optimal k (maximize coverage)
    optimal_idx = max(range(len(results)), key=lambda i: results[i]['coverage_pct'])
    optimal_k = results[optimal_idx]['k']

    print(f"\nOptimal k={optimal_k} ({results[optimal_idx]['coverage_pct']:.1f}% coverage)")

    return {
        'optimal_k': optimal_k,
        'results': results,
        'total_area_ha': total_area_ha,
    }


def run_kmeans(
    image: np.ndarray,
    n_clusters: int = 5,
) -> np.ndarray:
    """
    Run K-means clustering on image features, then label connected components.

    Args:
        image: Input image with shape (H, W, C), values in [0, 255] or [0, 1]
        n_clusters: Number of spectral classes to identify (default 5).
            More clusters = finer distinction between land cover types.

    Returns:
        Label array with shape (H, W) where each unique value is a segment ID.
        Each spatially connected region gets its own ID.
    """
    # Normalize to [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    if image.ndim == 2:
        image = image[:, :, None]

    h, w, c = image.shape

    # Flatten to (n_pixels, n_features)
    pixels = image.reshape(-1, c)

    # Handle NaN values - replace with 0 (will be background after clustering)
    nan_mask = np.any(np.isnan(pixels), axis=1)
    pixels = np.nan_to_num(pixels, nan=0.0)

    # Run K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pixels)

    # Reshape back to image
    cluster_map = cluster_labels.reshape(h, w)

    # Label connected components within each cluster
    # This splits each spectral class into separate spatial regions
    segments = np.zeros((h, w), dtype=np.int32)
    current_label = 1

    for cluster_id in range(n_clusters):
        mask = cluster_map == cluster_id
        labeled, n_regions = ndimage.label(mask)
        # Add to segments with offset
        segments[mask] = labeled[mask] + current_label - 1
        current_label += n_regions

    return segments


def segments_to_polygons(
    segments: np.ndarray,
    transform,
    crs,
) -> gpd.GeoDataFrame:
    """
    Convert segment label array to GeoDataFrame of polygons.

    Args:
        segments: Label array with shape (H, W)
        transform: Affine transform from raster coordinates to CRS
        crs: Coordinate reference system

    Returns:
        GeoDataFrame with polygon geometries
    """
    shapes_gen = features.shapes(
        segments.astype(np.int32),
        transform=transform,
    )

    records = []
    for geom, value in shapes_gen:
        if value == 0:  # Skip background
            continue
        records.append({
            'geometry': shape(geom),
            'segment_id': int(value),
        })

    if not records:
        return gpd.GeoDataFrame(columns=['geometry', 'segment_id'], crs=crs)

    gdf = gpd.GeoDataFrame(records, crs=crs)
    return gdf


def filter_polygons(
    gdf: gpd.GeoDataFrame,
    min_area_ha: float = 10,
    max_area_ha: float = 1500,
    min_compactness: float = 0.1,
) -> gpd.GeoDataFrame:
    """
    Filter polygons by area and shape metrics.

    Args:
        gdf: GeoDataFrame with polygon geometries
        min_area_ha: Minimum area in hectares
        max_area_ha: Maximum area in hectares
        min_compactness: Minimum compactness ratio (0-1). Compactness is
            4*pi*area/perimeter², where 1.0 = perfect circle, 0.785 = square.
            Default 0.3 filters out highly irregular/fragmented shapes.

    Returns:
        Filtered GeoDataFrame
    """
    if len(gdf) == 0:
        return gdf

    # Compute metrics (area in hectares, assuming metric CRS)
    gdf = gdf.copy()
    gdf['area_ha'] = gdf.geometry.area / 10000  # m² to hectares
    gdf['perimeter'] = gdf.geometry.length

    # Isoperimetric compactness: 4*pi*area/perimeter² (scale-invariant, 0-1 range)
    # Circle = 1.0, Square = pi/4 ≈ 0.785, more irregular = lower
    gdf['compactness'] = (4 * np.pi * gdf.geometry.area) / (gdf['perimeter'] ** 2)

    # Apply filters
    mask = (
        (gdf['area_ha'] >= min_area_ha) &
        (gdf['area_ha'] <= max_area_ha) &
        (gdf['compactness'] >= min_compactness)
    )

    gdf_filtered = gdf[mask].reset_index(drop=True)

    # Relabel with sequential paddock IDs (1, 2, 3, ...)
    gdf_filtered['paddock_id'] = range(1, len(gdf_filtered) + 1)
    gdf_filtered = gdf_filtered.drop(columns=['segment_id'])

    return gdf_filtered


def segment(
    query: Query,
    method: str = 'auto',
    n_clusters: int | str = 'auto',
    min_area_ha: float = 5,
    max_area_ha: float = 1500,
    min_compactness: float = 0.1,
    k_range: range = range(3, 12),
) -> None:
    """
    Main segmentation pipeline.

    Supports multiple segmentation methods:
    - 'watershed': Uses edge-based watershed (best for obvious paddocks)
    - 'kmeans': Uses K-means clustering (best for complex boundaries)
    - 'auto': Tries both and keeps the one with better coverage

    Args:
        query: Query object with paths
        method: Segmentation method ('auto', 'watershed', 'kmeans')
        n_clusters: Number of clusters for K-means (int or 'auto')
        min_area_ha: Minimum paddock area in hectares (default 10)
        max_area_ha: Maximum paddock area in hectares (default 1500)
        min_compactness: Minimum shape compactness 0-1 (default 0.1)
        k_range: Range of k values when n_clusters='auto' (default 3-11)
    """
    path_preseg_image = query.path_preseg_tif
    path_output_vector = query.path_polygons

    if not exists(path_preseg_image):
        raise FileNotFoundError(
            f"Presegmentation GeoTIFF not found at {path_preseg_image}. "
            "Run presegment(query) first."
        )

    # Load presegmented image
    data_xr = xr.open_dataarray(path_preseg_image)

    # Get image as numpy array (band, y, x) -> (y, x, band)
    image = data_xr.values
    if image.ndim == 3 and image.shape[0] <= 4:  # (band, y, x)
        image = np.moveaxis(image, 0, -1)

    # Get geospatial info
    transform = data_xr.rio.transform()
    crs = data_xr.rio.crs

    # Calculate total area for coverage metrics
    h, w = image.shape[:2]
    pixel_area = abs(transform.a * transform.e)
    total_area_ha = (h * w * pixel_area) / 10000

    def evaluate_segments(segments, label):
        """Helper to evaluate segmentation quality."""
        gdf = segments_to_polygons(segments, transform, crs)
        gdf_filtered = filter_polygons(
            gdf, min_area_ha=min_area_ha,
            max_area_ha=max_area_ha, min_compactness=min_compactness,
        )
        n_paddocks = len(gdf_filtered)
        area = gdf_filtered['area_ha'].sum() if n_paddocks > 0 else 0
        coverage = (area / total_area_ha) * 100

        # Score balances coverage with paddock count
        # Penalize results with too few paddocks (likely merged)
        # Target: 10+ paddocks is ideal, fewer is worse
        paddock_factor = min(1.0, n_paddocks / 10)
        score = coverage * paddock_factor

        print(f"{label}: {n_paddocks} paddocks, {coverage:.1f}% coverage (score={score:.1f})")
        return gdf_filtered, score

    best_gdf = None
    best_score = 0

    # Try watershed (good for obvious paddocks with clear edges)
    if method in ('auto', 'watershed'):
        print("\n--- Watershed segmentation ---")
        for pct in [10, 15, 20, 25]:
            segments_ws = run_watershed(image, marker_percentile=pct)
            gdf_ws, cov_ws = evaluate_segments(segments_ws, f"percentile={pct}")
            if cov_ws > best_score:
                best_score = cov_ws
                best_gdf = gdf_ws

    # Try K-means (good for complex boundaries)
    if method in ('auto', 'kmeans'):
        print("\n--- K-means segmentation ---")
        if n_clusters == 'auto':
            opt_result = find_optimal_k(
                image, transform, crs,
                k_range=k_range,
                min_area_ha=min_area_ha,
                max_area_ha=max_area_ha,
                min_compactness=min_compactness,
            )
            k = opt_result['optimal_k']
        else:
            k = n_clusters

        segments_km = run_kmeans(image, n_clusters=k)
        gdf_km, cov_km = evaluate_segments(segments_km, f"k={k}")
        if cov_km > best_score:
            best_score = cov_km
            best_gdf = gdf_km

    # Save best result
    if best_gdf is not None and len(best_gdf) > 0:
        print(f"\n==> Best: {len(best_gdf)} paddocks, {best_score:.1f}% coverage")
        best_gdf.to_file(path_output_vector, driver='GPKG')
    else:
        print("No valid paddocks found")
        gpd.GeoDataFrame(columns=['geometry', 'paddock_id'], crs=crs).to_file(
            path_output_vector, driver='GPKG'
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
