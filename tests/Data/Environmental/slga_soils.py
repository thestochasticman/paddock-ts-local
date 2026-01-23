"""Test slga_soils download."""
from PaddockTS.Data.Environmental import slga_soils
from tests.conftest import get_test_query, cleanup_test_data


def test_slga_soils():
    query = get_test_query()
    cleanup_test_data(query)

    print("\n=== Testing slga_soils ===")
    slga_soils(query, verbose=True)
    print("[done] slga_soils passed")


if __name__ == '__main__':
    test_slga_soils()
