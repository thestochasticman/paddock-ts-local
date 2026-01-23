"""
High-level orchestrator for PaddockSegmentation2 pipeline.

This module runs the full paddock segmentation workflow:
1. Download Sentinel-2 data (if not already present)
2. Compute spectral temporal features (NDVI, NDWI, edges)
3. Run watershed/K-means segmentation and filter polygons

No deep learning models required - uses classical image segmentation.
"""

from os.path import exists

from PaddockTS.PaddockSegmentation2._1_presegment import presegment
from PaddockTS.PaddockSegmentation2._2_segment import segment
from PaddockTS.Data.download_sentinel2 import download_sentinel2
from PaddockTS.query import Query


def get_paddocks(
    query: Query,
    method: str = 'auto',
    min_area_ha: float = 5,
    max_area_ha: float = 1500,
    min_compactness: float = 0.1,
    reload: bool = False,
) -> None:
    """
    Run the full paddock segmentation pipeline.

    Uses watershed + K-means segmentation (no SAM/deep learning required).

    Args:
        query: Query object specifying area of interest and time range
        method: Segmentation method
            - 'auto' (default): Tries watershed + K-means, picks best
            - 'watershed': Edge-based watershed (good for obvious paddocks)
            - 'kmeans': K-means clustering (good for complex boundaries)
        min_area_ha: Minimum paddock area in hectares (default 5)
        max_area_ha: Maximum paddock area in hectares (default 1500)
        min_compactness: Minimum shape compactness 0-1 (default 0.1).
            Circle=1.0, Square=0.785. Filters irregular/fragmented shapes.
        reload: If True, recompute all stages even if outputs exist

    Output files:
        - {query.stub_tmp_dir}/ds2.pkl: Sentinel-2 dataset
        - {query.stub_tmp_dir}/preseg.tif: Spectral temporal features GeoTIFF
        - {query.stub_tmp_dir}/polygons.gpkg: Filtered paddock polygons
    """
    # Stage 0: Download Sentinel-2 data
    if not exists(query.path_ds2) or reload:
        download_sentinel2(query)

    # Stage 1: Presegmentation (NDVI temporal features)
    presegment(query)

    # Stage 2: Segmentation (watershed + K-means) and polygon filtering
    segment(
        query,
        method=method,
        min_area_ha=min_area_ha,
        max_area_ha=max_area_ha,
        min_compactness=min_compactness,
    )


def test():
    from PaddockTS.query import get_example_query

    query = get_example_query()
    get_paddocks(query)
    print(f"Polygons: {query.path_polygons}")
    return exists(query.path_polygons)


if __name__ == '__main__':
    test()
