"""Test PaddockSegmentation2 presegmentation functions."""
import numpy as np
import xarray as xr
from PaddockTS.PaddockSegmentation2._1_presegment import (
    compute_ndvi,
    rescale_uint8,
    compute_ndvi_temporal_features,
)


def create_mock_dataset(height=20, width=20, n_times=12):
    """Create a mock Sentinel-2 dataset for testing."""
    # Create coordinate arrays
    y = np.linspace(-33.5, -33.4, height)
    x = np.linspace(148.3, 148.4, width)
    time = np.arange(n_times)

    # Create synthetic reflectance data (scaled by 10000 as in real data)
    # Red band: lower in vegetated areas
    red = np.random.randint(500, 2000, size=(height, width, n_times)).astype(np.float32)
    # NIR band: higher in vegetated areas
    nir = np.random.randint(2000, 5000, size=(height, width, n_times)).astype(np.float32)

    ds = xr.Dataset(
        {
            'nbart_red': (['y', 'x', 'time'], red),
            'nbart_nir_1': (['y', 'x', 'time'], nir),
        },
        coords={
            'y': y,
            'x': x,
            'time': time,
        }
    )

    return ds


def test_compute_ndvi_shape():
    """Test NDVI computation returns correct shape."""
    print("\n=== Testing compute_ndvi shape ===")
    ds = create_mock_dataset(height=10, width=10, n_times=6)
    ndvi = compute_ndvi(ds)

    assert ndvi.shape == (10, 10, 6), f"Expected (10, 10, 6), got {ndvi.shape}"
    assert ndvi.dtype == np.float32
    print("[done] compute_ndvi shape passed")


def test_compute_ndvi_range():
    """Test NDVI values are in valid range [-1, 1]."""
    print("\n=== Testing compute_ndvi range ===")
    ds = create_mock_dataset()
    ndvi = compute_ndvi(ds)

    # Filter out NaN values
    valid = ndvi[~np.isnan(ndvi)]
    assert valid.min() >= -1.0, "NDVI should be >= -1"
    assert valid.max() <= 1.0, "NDVI should be <= 1"
    print("[done] compute_ndvi range passed")


def test_compute_ndvi_formula():
    """Test NDVI formula: (NIR - Red) / (NIR + Red)."""
    print("\n=== Testing compute_ndvi formula ===")
    # Create dataset with known values
    y = np.array([0.0])
    x = np.array([0.0])
    time = np.array([0])

    # Red = 2000 (scaled), NIR = 4000 (scaled)
    # After scaling: red = 0.2, nir = 0.4
    # NDVI = (0.4 - 0.2) / (0.4 + 0.2) = 0.2 / 0.6 = 0.333...
    ds = xr.Dataset(
        {
            'nbart_red': (['y', 'x', 'time'], np.array([[[2000.0]]])),
            'nbart_nir_1': (['y', 'x', 'time'], np.array([[[4000.0]]])),
        },
        coords={'y': y, 'x': x, 'time': time}
    )

    ndvi = compute_ndvi(ds)
    expected = (0.4 - 0.2) / (0.4 + 0.2)

    assert np.isclose(ndvi[0, 0, 0], expected, atol=0.001), \
        f"Expected {expected}, got {ndvi[0, 0, 0]}"
    print("[done] compute_ndvi formula passed")


def test_compute_ndvi_zero_handling():
    """Test NDVI handles zero values (invalid data)."""
    print("\n=== Testing compute_ndvi zero handling ===")
    y = np.array([0.0])
    x = np.array([0.0])
    time = np.array([0, 1])

    # First timestep: valid data, second: zero (invalid)
    ds = xr.Dataset(
        {
            'nbart_red': (['y', 'x', 'time'], np.array([[[2000.0, 0.0]]])),
            'nbart_nir_1': (['y', 'x', 'time'], np.array([[[4000.0, 0.0]]])),
        },
        coords={'y': y, 'x': x, 'time': time}
    )

    ndvi = compute_ndvi(ds)

    assert not np.isnan(ndvi[0, 0, 0]), "Valid data should not be NaN"
    assert np.isnan(ndvi[0, 0, 1]), "Zero data should become NaN"
    print("[done] compute_ndvi zero handling passed")


def test_rescale_uint8_basic():
    """Test rescaling to uint8 [0, 255]."""
    print("\n=== Testing rescale_uint8 basic ===")
    im = np.array([[[0.0], [0.5], [1.0]]], dtype=np.float32)
    result = rescale_uint8(im)

    assert result.dtype == np.uint8
    assert result[0, 0, 0] == 0, "Min should be 0"
    assert result[0, 2, 0] == 255, "Max should be 255"
    assert result[0, 1, 0] == 127 or result[0, 1, 0] == 128, "Middle should be ~127-128"
    print("[done] rescale_uint8 basic passed")


def test_rescale_uint8_multiband():
    """Test rescaling preserves multiple bands."""
    print("\n=== Testing rescale_uint8 multiband ===")
    im = np.random.rand(10, 10, 3).astype(np.float32)
    result = rescale_uint8(im)

    assert result.shape == (10, 10, 3)
    assert result.dtype == np.uint8
    print("[done] rescale_uint8 multiband passed")


def test_rescale_uint8_handles_nan():
    """Test rescaling handles NaN values."""
    print("\n=== Testing rescale_uint8 NaN handling ===")
    im = np.array([[[0.0, np.nan, 1.0]]], dtype=np.float32).reshape(1, 3, 1)
    result = rescale_uint8(im)

    assert result.dtype == np.uint8
    assert result[0, 0, 0] == 0, "Min should be 0"
    assert result[0, 2, 0] == 255, "Max should be 255"
    assert result[0, 1, 0] == 0, "NaN should become 0"
    print("[done] rescale_uint8 NaN handling passed")


def test_rescale_uint8_constant():
    """Test rescaling handles constant array."""
    print("\n=== Testing rescale_uint8 constant ===")
    im = np.full((5, 5, 1), 0.5, dtype=np.float32)
    result = rescale_uint8(im)

    assert result.dtype == np.uint8
    assert np.all(result == 0), "Constant array should become zeros"
    print("[done] rescale_uint8 constant passed")


def test_rescale_uint8_2d():
    """Test rescaling handles 2D input."""
    print("\n=== Testing rescale_uint8 2D input ===")
    im = np.array([[0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    result = rescale_uint8(im)

    assert result.shape == (2, 2, 1), "2D should expand to 3D"
    assert result.dtype == np.uint8
    print("[done] rescale_uint8 2D input passed")


def test_compute_ndvi_temporal_features():
    """Test full NDVI temporal feature extraction."""
    print("\n=== Testing compute_ndvi_temporal_features ===")
    ds = create_mock_dataset(height=15, width=15, n_times=10)
    features = compute_ndvi_temporal_features(ds)

    assert features.shape == (15, 15, 3), f"Expected (15, 15, 3), got {features.shape}"
    assert features.dtype == np.float32
    # Features should be normalized
    assert features.min() >= 0.0
    assert features.max() <= 1.0
    print("[done] compute_ndvi_temporal_features passed")


def test_all():
    """Run all presegment tests."""
    print("=" * 50)
    print("Running PaddockSegmentation2 presegment tests...")
    print("=" * 50)

    test_compute_ndvi_shape()
    test_compute_ndvi_range()
    test_compute_ndvi_formula()
    test_compute_ndvi_zero_handling()
    test_rescale_uint8_basic()
    test_rescale_uint8_multiband()
    test_rescale_uint8_handles_nan()
    test_rescale_uint8_constant()
    test_rescale_uint8_2d()
    test_compute_ndvi_temporal_features()

    print("\n" + "=" * 50)
    print("All presegment tests passed!")
    print("=" * 50)


if __name__ == '__main__':
    test_all()
