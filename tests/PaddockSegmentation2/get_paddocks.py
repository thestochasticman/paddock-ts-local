"""Integration test for PaddockSegmentation2 pipeline."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from os.path import exists
from tests.conftest import get_test_query, cleanup_test_data
from PaddockTS.PaddockSegmentation2.get_paddocks import get_paddocks
from PaddockTS.PaddockSegmentation2._1_presegment import presegment
from PaddockTS.PaddockSegmentation2._2_segment import segment
import geopandas as gpd


def test_get_paddocks_integration():
    """
    Integration test for full paddock segmentation pipeline.

    Note: This test requires Sentinel-2 data to be downloaded,
    which needs network access. If data already exists, it will be reused.
    """
    print("\n=== Testing get_paddocks integration ===")
    print("Note: This test requires network access for Sentinel-2 download")

    query = get_test_query()
    print(f"Query stub: {query.stub}")
    print(f"Location: ({query.lat}, {query.lon})")
    print(f"Date range: {query.datetime}")

    # Run full pipeline
    get_paddocks(
        query,
        n_segments=50,  # Fewer segments for faster test
        compactness=10.0,
        min_area_ha=1,  # Lower threshold for small test area
        max_area_ha=1500,
        max_perim_area_ratio=50,
    )

    # Verify outputs exist
    assert exists(query.path_ds2), f"Sentinel-2 data not found: {query.path_ds2}"
    print(f"[ok] Sentinel-2 data: {query.path_ds2}")

    assert exists(query.path_preseg_tif), f"Preseg GeoTIFF not found: {query.path_preseg_tif}"
    print(f"[ok] Preseg GeoTIFF: {query.path_preseg_tif}")

    assert exists(query.path_polygons), f"Polygons not found: {query.path_polygons}"
    print(f"[ok] Polygons: {query.path_polygons}")

    # Verify polygon output
    gdf = gpd.read_file(query.path_polygons)
    print(f"[ok] Found {len(gdf)} paddock polygons")

    assert len(gdf) > 0, "Should have at least one polygon"
    assert 'area_ha' in gdf.columns, "Polygons should have area_ha column"
    assert gdf.crs is not None, "Polygons should have CRS"

    print("[done] get_paddocks integration passed")
    return gdf


def test_presegment_only():
    """Test presegmentation stage in isolation (requires ds2.pkl)."""
    print("\n=== Testing presegment only ===")

    query = get_test_query()

    if not exists(query.path_ds2):
        print("[skip] Sentinel-2 data not found, skipping presegment test")
        print("       Run test_get_paddocks_integration first to download data")
        return None

    # Remove existing preseg to force recomputation
    import os
    if exists(query.path_preseg_tif):
        os.remove(query.path_preseg_tif)

    # Run presegmentation
    result = presegment(query)

    assert exists(query.path_preseg_tif), "Preseg GeoTIFF should be created"
    assert result is not None, "presegment should return DataArray"

    print(f"[ok] Preseg shape: {result.shape}")
    print("[done] presegment only passed")
    return result


def test_segment_only():
    """Test segmentation stage in isolation (requires preseg.tif)."""
    print("\n=== Testing segment only ===")

    query = get_test_query()

    if not exists(query.path_preseg_tif):
        print("[skip] Preseg GeoTIFF not found, skipping segment test")
        print("       Run test_presegment_only first")
        return None

    # Remove existing polygons to force recomputation
    import os
    if exists(query.path_polygons):
        os.remove(query.path_polygons)

    # Run segmentation
    segment(
        query,
        n_segments=50,
        compactness=10.0,
        min_area_ha=1,
        max_area_ha=1500,
    )

    assert exists(query.path_polygons), "Polygons should be created"

    gdf = gpd.read_file(query.path_polygons)
    print(f"[ok] Found {len(gdf)} polygons")
    print("[done] segment only passed")
    return gdf


def test_reload_flag():
    """Test that reload=True forces recomputation."""
    print("\n=== Testing reload flag ===")

    query = get_test_query()

    if not exists(query.path_ds2):
        print("[skip] No existing data to test reload flag")
        return

    import os
    from datetime import datetime

    # Get modification times before
    preseg_mtime_before = None
    if exists(query.path_preseg_tif):
        preseg_mtime_before = os.path.getmtime(query.path_preseg_tif)

    # Run with reload=True
    get_paddocks(query, n_segments=30, reload=True)

    # Check that files were regenerated
    if preseg_mtime_before is not None:
        preseg_mtime_after = os.path.getmtime(query.path_preseg_tif)
        assert preseg_mtime_after > preseg_mtime_before, "Preseg should be regenerated"
        print("[ok] Preseg was regenerated")

    print("[done] reload flag passed")


def test_all():
    """Run all integration tests."""
    print("=" * 60)
    print("Running PaddockSegmentation2 integration tests...")
    print("=" * 60)
    print("\nWARNING: Integration tests require network access")
    print("         and may download Sentinel-2 data (~50-100MB)")
    print("=" * 60)

    # Run tests in order (each depends on previous)
    test_get_paddocks_integration()
    test_presegment_only()
    test_segment_only()
    test_reload_flag()

    print("\n" + "=" * 60)
    print("All integration tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_all()
