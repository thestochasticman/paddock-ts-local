"""
Stage 2: Contour-based polygon extraction.

Key improvement over PaddockSegmentation3:
- Uses cv2.findContours to extract polygons directly from cluster labels
- No watershed needed - simpler and cleaner boundaries
- cv2.approxPolyDP simplifies jagged edges
"""

from os.path import exists

import cv2
import numpy as np
import geopandas as gpd
import rioxarray
import xarray as xr
from shapely.geometry import Polygon
from scipy import ndimage

from PaddockTS.query import Query


def labels_to_polygons(
    labels: np.ndarray,
    transform,
    crs,
    epsilon_factor: float = 0.005,
) -> gpd.GeoDataFrame:
    """
    Extract polygons from cluster labels using cv2.findContours.

    Args:
        labels: Cluster label array (H, W)
        transform: Affine transform for georeferencing
        crs: Coordinate reference system
        epsilon_factor: Simplification factor for approxPolyDP (fraction of perimeter)

    Returns:
        GeoDataFrame with one polygon per connected region
    """
    unique_labels = np.unique(labels)
    records = []
    segment_id = 0

    for label in unique_labels:
        if label == 0:  # Skip background
            continue

        # Create binary mask for this cluster
        mask = (labels == label).astype(np.uint8)

        # Find contours
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Morphological smoothing before contour extraction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise

        # Re-extract contours from smoothed mask
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if len(contour) < 3:  # Need at least 3 points for a polygon
                continue

            # Convert pixel coordinates to geographic coordinates
            coords = []
            for pt in contour.reshape(-1, 2):
                x_geo = transform.c + pt[0] * transform.a
                y_geo = transform.f + pt[1] * transform.e
                coords.append((x_geo, y_geo))

            # Close the polygon
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            try:
                poly = Polygon(coords)
                if poly.is_valid and poly.area > 0:
                    segment_id += 1
                    records.append({
                        'geometry': poly,
                        'segment_id': segment_id,
                        'cluster_label': int(label),
                    })
            except Exception:
                continue

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
    gdf_filtered = gdf_filtered.drop(columns=['segment_id', 'cluster_label'])

    return gdf_filtered


def segment(
    query: Query,
    min_area_ha: float = 5,
    max_area_ha: float = 1500,
    min_compactness: float = 0.1,
    epsilon_factor: float = 0.005,
) -> None:
    """
    Extract paddock polygons using cv2.findContours.

    Args:
        query: Query object with paths
        min_area_ha: Minimum paddock area in hectares
        max_area_ha: Maximum paddock area in hectares
        min_compactness: Minimum shape compactness 0-1
        epsilon_factor: Polygon simplification factor (default 0.005)
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

    print(f"\n=== Contour-based Polygon Extraction ===")

    # Get cluster labels from band 0 (denormalize from 0-255 to cluster IDs)
    labels_band = image[:, :, 0] if image.ndim == 3 else image
    if labels_band.dtype == np.uint8:
        # Quantize back to cluster labels
        n_unique = len(np.unique(labels_band))
        labels = (labels_band.astype(np.float32) / 255.0 * (n_unique - 1)).astype(np.int32)
    else:
        labels = labels_band.astype(np.int32)

    # Label connected components within each cluster
    final_labels = np.zeros_like(labels, dtype=np.int32)
    current_label = 0
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        labeled, n_components = ndimage.label(mask)
        labeled[labeled > 0] += current_label
        final_labels[mask] = labeled[mask]
        current_label += n_components

    print(f"Found {current_label} connected regions across {len(np.unique(labels))} clusters")

    # Extract polygons using contours
    gdf = labels_to_polygons(final_labels, transform, crs, epsilon_factor)
    print(f"Extracted {len(gdf)} polygons")

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
