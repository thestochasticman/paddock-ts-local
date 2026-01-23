"""Test download_all function."""
from PaddockTS.Data.download import download_all
from tests.conftest import get_test_query, cleanup_test_data


def test_download_all():
    """Test parallel download of all data (Sentinel-2 + Environmental)."""
    query = get_test_query()
    cleanup_test_data(query)

    print("\n=== Testing download_all (Sentinel-2 + Environmental) ===")
    results = download_all(query, verbose=True)

    assert 'sentinel2' in results, "Missing 'sentinel2' result"
    assert 'environmental' in results, "Missing 'environmental' result"
    print("[done] download_all passed")

    return results


if __name__ == '__main__':
    print("=" * 50)
    print("Testing download_all...")
    print("=" * 50)

    test_download_all()

    print("\n" + "=" * 50)
    print("download_all test passed!")
    print("=" * 50)
