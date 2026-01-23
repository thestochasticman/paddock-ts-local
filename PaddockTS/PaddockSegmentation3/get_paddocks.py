"""
High-level orchestrator for PaddockSegmentation3 pipeline.

This module runs the full paddock segmentation workflow:
1. Download Sentinel-2 data (if not already present)
2. Time series K-means clustering (cluster by phenology)
3. Watershed segmentation on cluster edges

Key difference from PaddockSegmentation2:
- Clusters pixels by their full NDVI time series (phenological signature)
- Uses cluster boundaries as input for watershed
"""

from os.path import exists

from PaddockTS.PaddockSegmentation3._1_presegment import presegment
from PaddockTS.PaddockSegmentation3._2_segment import segment
from PaddockTS.Data.download_sentinel2 import download_sentinel2
from PaddockTS.query import Query


def get_paddocks(
    query: Query,
    n_clusters: int | str = 'auto',
    min_area_ha: float = 5,
    max_area_ha: float = 1500,
    min_compactness: float = 0.1,
    marker_percentile: float = 25,
    k_range: range = range(3, 16),
    scoring: str = 'coverage',
    reload: bool = False,
) -> None:
    """
    Run the full paddock segmentation pipeline.

    Uses time series K-means + watershed segmentation.

    Args:
        query: Query object specifying area of interest and time range
        n_clusters: Number of K-means clusters, or 'auto' to find optimal (default)
        min_area_ha: Minimum paddock area in hectares (default 5)
        max_area_ha: Maximum paddock area in hectares (default 1500)
        min_compactness: Minimum shape compactness 0-1 (default 0.1)
        marker_percentile: Percentile for watershed seed markers (default 25)
        k_range: Range of k values to try when n_clusters='auto' (default 4-15)
        scoring: Scoring method for optimal k - 'coverage', 'silhouette', or 'combined'
        reload: If True, recompute all stages even if outputs exist

    Output files:
        - {query.stub_tmp_dir}/ds2.pkl: Sentinel-2 dataset
        - {query.stub_tmp_dir}/preseg.tif: Cluster labels + edges GeoTIFF
        - {query.stub_tmp_dir}/polygons.gpkg: Filtered paddock polygons
    """
    # Stage 0: Download Sentinel-2 data
    if not exists(query.path_ds2) or reload:
        download_sentinel2(query)

    # Stage 1: Time series K-means clustering (auto-optimizes k by default)
    presegment(
        query,
        n_clusters=n_clusters,
        min_area_ha=min_area_ha,
        max_area_ha=max_area_ha,
        min_compactness=min_compactness,
        k_range=k_range,
        scoring=scoring,
    )

    # Stage 2: Watershed on cluster edges
    segment(
        query,
        min_area_ha=min_area_ha,
        max_area_ha=max_area_ha,
        min_compactness=min_compactness,
        marker_percentile=marker_percentile,
    )


def test():
    from PaddockTS.query import get_example_query

    query = get_example_query()
    get_paddocks(query)
    print(f"Polygons: {query.path_polygons}")
    return exists(query.path_polygons)


if __name__ == '__main__':
    test()
