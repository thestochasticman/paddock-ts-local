"""Test PaddockSegmentation2 utility functions."""
import numpy as np
from PaddockTS.PaddockSegmentation2.utils import (
    normalize,
    completion,
    compute_temporal_features,
)


def test_normalize_basic():
    """Test basic normalization to [0, 1] range."""
    print("\n=== Testing normalize basic ===")
    arr = np.array([0, 50, 100], dtype=np.float32)
    result = normalize(arr)

    assert result.min() == 0.0, "Min should be 0"
    assert result.max() == 1.0, "Max should be 1"
    assert result[1] == 0.5, "Middle value should be 0.5"
    print("[done] normalize basic passed")


def test_normalize_constant():
    """Test normalization of constant array."""
    print("\n=== Testing normalize constant array ===")
    arr = np.array([5, 5, 5], dtype=np.float32)
    result = normalize(arr)

    assert np.all(result == 0), "Constant array should normalize to zeros"
    print("[done] normalize constant passed")


def test_normalize_with_nan():
    """Test normalization handles NaN values."""
    print("\n=== Testing normalize with NaN ===")
    arr = np.array([0, np.nan, 100], dtype=np.float32)
    result = normalize(arr)

    assert result[0] == 0.0, "Min should normalize to 0"
    assert result[2] == 1.0, "Max should normalize to 1"
    assert np.isnan(result[1]), "NaN should remain NaN"
    print("[done] normalize with NaN passed")


def test_normalize_2d():
    """Test normalization of 2D array."""
    print("\n=== Testing normalize 2D ===")
    arr = np.array([[0, 50], [100, 75]], dtype=np.float32)
    result = normalize(arr)

    assert result.shape == arr.shape, "Shape should be preserved"
    assert result.min() == 0.0
    assert result.max() == 1.0
    print("[done] normalize 2D passed")


def test_completion_forward_fill():
    """Test forward-fill of NaN values."""
    print("\n=== Testing completion forward fill ===")
    # Shape (H, W, T) = (1, 1, 5)
    arr = np.array([[[0.5, np.nan, np.nan, 0.8, np.nan]]], dtype=np.float32)
    result = completion(arr)

    assert result[0, 0, 0] == 0.5, "First value unchanged"
    assert result[0, 0, 1] == 0.5, "Second value forward-filled"
    assert result[0, 0, 2] == 0.5, "Third value forward-filled"
    assert result[0, 0, 3] == 0.8, "Fourth value unchanged"
    assert result[0, 0, 4] == 0.8, "Fifth value forward-filled"
    print("[done] completion forward fill passed")


def test_completion_initial_nan():
    """Test completion fills initial NaN with mean."""
    print("\n=== Testing completion initial NaN ===")
    # Initial NaN should be filled with mean of valid values
    arr = np.array([[[np.nan, np.nan, 0.4, 0.6, 0.8]]], dtype=np.float32)
    result = completion(arr)

    # Mean of [0.4, 0.6, 0.8] = 0.6
    expected_mean = 0.6
    assert np.isclose(result[0, 0, 0], expected_mean, atol=0.01), "Initial NaN filled with mean"
    assert np.isclose(result[0, 0, 1], expected_mean, atol=0.01), "Second NaN filled with mean"
    print("[done] completion initial NaN passed")


def test_completion_all_nan():
    """Test completion handles all-NaN series."""
    print("\n=== Testing completion all NaN ===")
    arr = np.array([[[np.nan, np.nan, np.nan]]], dtype=np.float32)
    result = completion(arr)

    assert np.all(result == 0), "All-NaN series should become zeros"
    print("[done] completion all NaN passed")


def test_completion_no_nan():
    """Test completion leaves valid data unchanged."""
    print("\n=== Testing completion no NaN ===")
    arr = np.array([[[0.1, 0.2, 0.3, 0.4]]], dtype=np.float32)
    result = completion(arr)

    np.testing.assert_array_almost_equal(result, arr)
    print("[done] completion no NaN passed")


def test_compute_temporal_features_shape():
    """Test temporal features output shape."""
    print("\n=== Testing compute_temporal_features shape ===")
    # Create (H, W, T) = (10, 10, 24) array
    ndvi = np.random.rand(10, 10, 24).astype(np.float32)
    result = compute_temporal_features(ndvi)

    assert result.shape == (10, 10, 3), f"Expected (10, 10, 3), got {result.shape}"
    assert result.dtype == np.float32
    print("[done] compute_temporal_features shape passed")


def test_compute_temporal_features_normalized():
    """Test temporal features are normalized to [0, 1]."""
    print("\n=== Testing compute_temporal_features normalized ===")
    ndvi = np.random.rand(10, 10, 24).astype(np.float32)
    result = compute_temporal_features(ndvi)

    assert result.min() >= 0.0, "Min should be >= 0"
    assert result.max() <= 1.0, "Max should be <= 1"
    print("[done] compute_temporal_features normalized passed")


def test_compute_temporal_features_distinct_paddocks():
    """Test features distinguish different temporal patterns."""
    print("\n=== Testing compute_temporal_features distinct paddocks ===")
    # Create two distinct regions
    ndvi = np.zeros((4, 4, 12), dtype=np.float32)

    # Region 1 (top-left): stable high NDVI
    ndvi[:2, :2, :] = 0.7

    # Region 2 (bottom-right): variable NDVI (seasonal)
    for t in range(12):
        ndvi[2:, 2:, t] = 0.3 + 0.4 * np.sin(2 * np.pi * t / 12)

    result = compute_temporal_features(ndvi)

    # Mean feature (band 0)
    mean_region1 = result[:2, :2, 0].mean()
    mean_region2 = result[2:, 2:, 0].mean()

    # Std feature (band 1) - region 2 should have higher std
    std_region1 = result[:2, :2, 1].mean()
    std_region2 = result[2:, 2:, 1].mean()

    assert std_region2 > std_region1, "Variable region should have higher std feature"
    print("[done] compute_temporal_features distinct paddocks passed")


def test_all():
    """Run all utils tests."""
    print("=" * 50)
    print("Running PaddockSegmentation2 utils tests...")
    print("=" * 50)

    test_normalize_basic()
    test_normalize_constant()
    test_normalize_with_nan()
    test_normalize_2d()
    test_completion_forward_fill()
    test_completion_initial_nan()
    test_completion_all_nan()
    test_completion_no_nan()
    test_compute_temporal_features_shape()
    test_compute_temporal_features_normalized()
    test_compute_temporal_features_distinct_paddocks()

    print("\n" + "=" * 50)
    print("All utils tests passed!")
    print("=" * 50)


if __name__ == '__main__':
    test_all()
