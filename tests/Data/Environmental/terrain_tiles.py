"""Test terrain_tiles download."""
from PaddockTS.Data.Environmental import terrain_tiles
from tests.conftest import get_test_query, cleanup_test_data


def test_terrain_tiles():
    query = get_test_query()
    cleanup_test_data(query)

    print("\n=== Testing terrain_tiles ===")
    ds = terrain_tiles(query, verbose=True)
    assert ds is not None, "terrain_tiles returned None"
    assert 'terrain' in ds.data_vars, "Missing 'terrain' variable"
    print("[done] terrain_tiles passed")
    return ds


if __name__ == '__main__':
    test_terrain_tiles()
