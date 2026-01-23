"""Test download_environmental parallel download."""
from PaddockTS.Data.download_environmental import download_environmental
from tests.conftest import get_test_query, cleanup_test_data


def test_download_environmental():
    query = get_test_query()
    cleanup_test_data(query)

    print("\n=== Testing download_environmental (parallel) ===")
    results = download_environmental(query, verbose=True)
    assert 'terrain' in results, "Missing 'terrain' result"
    assert 'silo_daily' in results, "Missing 'silo_daily' result"
    assert 'ozwald_8day' in results, "Missing 'ozwald_8day' result"
    assert 'daesim_forcing' in results, "Missing 'daesim_forcing' result"
    assert 'daesim_soils' in results, "Missing 'daesim_soils' result"
    print("[done] download_environmental passed")
    return results


if __name__ == '__main__':
    test_download_environmental()
