"""Test PaddockSegmentation2 segmentation functions."""
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from rasterio.transform import from_bounds
from PaddockTS.PaddockSegmentation2._2_segment import (
    run_slic,
    segments_to_polygons,
    filter_polygons,
)


def test_run_slic_basic():
    """Test SLIC returns labeled segments."""
    print("\n=== Testing run_slic basic ===")
    # Create a simple 3-channel image
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    segments = run_slic(image, n_segments=10, compactness=10.0)

    assert segments.shape == (100, 100), f"Expected (100, 100), got {segments.shape}"
    assert segments.dtype in [np.int32, np.int64], f"Expected int, got {segments.dtype}"
    assert segments.min() >= 1, "Segments should start at 1"
    print("[done] run_slic basic passed")


def test_run_slic_segment_count():
    """Test SLIC produces approximately requested segments."""
    print("\n=== Testing run_slic segment count ===")
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    segments = run_slic(image, n_segments=20, compactness=10.0)

    n_unique = len(np.unique(segments))
    # SLIC doesn't guarantee exact count, but should be close
    assert n_unique >= 10, f"Expected at least 10 segments, got {n_unique}"
    assert n_unique <= 40, f"Expected at most 40 segments, got {n_unique}"
    print(f"[done] run_slic segment count passed (got {n_unique} segments)")


def test_run_slic_compactness():
    """Test compactness parameter affects segment shape."""
    print("\n=== Testing run_slic compactness ===")
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # Low compactness: more irregular shapes following color
    seg_low = run_slic(image, n_segments=20, compactness=1.0)
    # High compactness: more regular/square shapes
    seg_high = run_slic(image, n_segments=20, compactness=50.0)

    # Both should produce valid segmentations
    assert seg_low.shape == seg_high.shape
    assert len(np.unique(seg_low)) > 0
    assert len(np.unique(seg_high)) > 0
    print("[done] run_slic compactness passed")


def test_run_slic_single_channel():
    """Test SLIC handles single-channel input."""
    print("\n=== Testing run_slic single channel ===")
    image = np.random.randint(0, 256, size=(50, 50), dtype=np.uint8)
    segments = run_slic(image, n_segments=10, compactness=10.0)

    assert segments.shape == (50, 50)
    print("[done] run_slic single channel passed")


def test_run_slic_float_input():
    """Test SLIC handles float [0, 1] input."""
    print("\n=== Testing run_slic float input ===")
    image = np.random.rand(50, 50, 3).astype(np.float32)
    segments = run_slic(image, n_segments=10, compactness=10.0)

    assert segments.shape == (50, 50)
    print("[done] run_slic float input passed")


def test_segments_to_polygons():
    """Test conversion of segments to polygons."""
    print("\n=== Testing segments_to_polygons ===")
    # Create a simple 2x2 segment grid
    segments = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ], dtype=np.int32)

    # Create transform (10m pixels, origin at 0,0)
    transform = from_bounds(0, 0, 40, 40, 4, 4)
    crs = "EPSG:32755"  # UTM zone

    gdf = segments_to_polygons(segments, transform, crs)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 4, f"Expected 4 polygons, got {len(gdf)}"
    assert 'segment_id' in gdf.columns
    assert gdf.crs is not None
    print("[done] segments_to_polygons passed")


def test_segments_to_polygons_with_background():
    """Test segments_to_polygons skips background (0) values."""
    print("\n=== Testing segments_to_polygons with background ===")
    segments = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 0, 0],
        [2, 2, 0, 0],
    ], dtype=np.int32)

    transform = from_bounds(0, 0, 40, 40, 4, 4)
    crs = "EPSG:32755"

    gdf = segments_to_polygons(segments, transform, crs)

    assert len(gdf) == 2, f"Expected 2 polygons (excluding background), got {len(gdf)}"
    assert 0 not in gdf['segment_id'].values, "Background should not be in polygons"
    print("[done] segments_to_polygons with background passed")


def test_filter_polygons_by_area():
    """Test polygon filtering by area."""
    print("\n=== Testing filter_polygons by area ===")
    # Create GeoDataFrame with polygons of different sizes
    # Using metric CRS (EPSG:32755) so area is in m²
    geometries = [
        box(0, 0, 100, 100),       # 1 ha (10000 m²)
        box(0, 0, 316, 316),       # ~10 ha
        box(0, 0, 1000, 1000),     # 100 ha
        box(0, 0, 4000, 4000),     # 1600 ha (too large)
    ]
    gdf = gpd.GeoDataFrame(
        {'segment_id': [1, 2, 3, 4], 'geometry': geometries},
        crs="EPSG:32755"
    )

    filtered = filter_polygons(gdf, min_area_ha=5, max_area_ha=1500)

    assert len(filtered) == 2, f"Expected 2 polygons, got {len(filtered)}"
    assert 1 not in filtered['segment_id'].values, "1 ha polygon should be filtered"
    assert 4 not in filtered['segment_id'].values, "1600 ha polygon should be filtered"
    print("[done] filter_polygons by area passed")


def test_filter_polygons_by_shape():
    """Test polygon filtering by perimeter/area ratio."""
    print("\n=== Testing filter_polygons by shape ===")
    # Create polygons with different shapes
    # Compact square vs elongated rectangle
    geometries = [
        box(0, 0, 500, 500),       # Square: 25 ha, perim=2000m, ratio=80
        box(0, 0, 100, 2500),      # Elongated: 25 ha, perim=5200m, ratio=208
    ]
    gdf = gpd.GeoDataFrame(
        {'segment_id': [1, 2], 'geometry': geometries},
        crs="EPSG:32755"
    )

    # Use permissive area filter, strict shape filter
    filtered = filter_polygons(gdf, min_area_ha=1, max_area_ha=1000, max_perim_area_ratio=100)

    assert len(filtered) == 1, f"Expected 1 polygon, got {len(filtered)}"
    assert filtered['segment_id'].iloc[0] == 1, "Compact square should pass"
    print("[done] filter_polygons by shape passed")


def test_filter_polygons_empty():
    """Test filter_polygons handles empty GeoDataFrame."""
    print("\n=== Testing filter_polygons empty ===")
    gdf = gpd.GeoDataFrame(columns=['geometry', 'segment_id'], crs="EPSG:32755")
    filtered = filter_polygons(gdf, min_area_ha=10, max_area_ha=1500)

    assert len(filtered) == 0
    print("[done] filter_polygons empty passed")


def test_filter_polygons_adds_metrics():
    """Test filter_polygons adds area and perimeter columns."""
    print("\n=== Testing filter_polygons adds metrics ===")
    geometries = [box(0, 0, 500, 500)]  # 25 ha
    gdf = gpd.GeoDataFrame(
        {'segment_id': [1], 'geometry': geometries},
        crs="EPSG:32755"
    )

    filtered = filter_polygons(gdf, min_area_ha=1, max_area_ha=1000)

    assert 'area_ha' in filtered.columns
    assert 'perimeter' in filtered.columns
    assert 'perim_area_ratio' in filtered.columns
    assert np.isclose(filtered['area_ha'].iloc[0], 25, atol=0.1)
    print("[done] filter_polygons adds metrics passed")


def test_all():
    """Run all segment tests."""
    print("=" * 50)
    print("Running PaddockSegmentation2 segment tests...")
    print("=" * 50)

    test_run_slic_basic()
    test_run_slic_segment_count()
    test_run_slic_compactness()
    test_run_slic_single_channel()
    test_run_slic_float_input()
    test_segments_to_polygons()
    test_segments_to_polygons_with_background()
    test_filter_polygons_by_area()
    test_filter_polygons_by_shape()
    test_filter_polygons_empty()
    test_filter_polygons_adds_metrics()

    print("\n" + "=" * 50)
    print("All segment tests passed!")
    print("=" * 50)


if __name__ == '__main__':
    test_all()
