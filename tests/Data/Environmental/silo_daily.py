"""Test silo_daily download."""
from PaddockTS.Data.Environmental import silo_daily
from tests.conftest import get_test_query, cleanup_test_data


def test_silo_daily():
    query = get_test_query()
    cleanup_test_data(query)

    print("\n=== Testing silo_daily ===")
    ds = silo_daily(
        query,
        variables=['radiation', 'vp', 'max_temp', 'min_temp', 'daily_rain', 'et_morton_actual', 'et_morton_potential'],
        save_netcdf=True,
        save_json=False,
        plot=False,
        verbose=True
    )
    assert ds is not None, "silo_daily returned None"
    assert 'radiation' in ds.data_vars, "Missing 'radiation' variable"
    print("[done] silo_daily passed")
    return ds


if __name__ == '__main__':
    test_silo_daily()
