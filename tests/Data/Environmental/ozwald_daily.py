"""Test ozwald_daily download."""
from PaddockTS.Data.Environmental import ozwald_daily
from tests.conftest import get_test_query, cleanup_test_data


def test_ozwald_daily():
    query = get_test_query()
    cleanup_test_data(query)

    print("\n=== Testing ozwald_daily ===")

    # Pg (rainfall)
    ds_pg = ozwald_daily(
        query,
        variables=['Pg'],
        save_netcdf=True,
        save_json=False,
        plot=False,
        verbose=True
    )
    assert 'Pg' in ds_pg.data_vars, "Missing 'Pg' variable"

    # Tmax, Tmin (temperature)
    ds_tmax = ozwald_daily(
        query,
        variables=['Tmax', 'Tmin'],
        save_netcdf=True,
        save_json=False,
        plot=False,
        verbose=True
    )
    assert 'Tmax' in ds_tmax.data_vars, "Missing 'Tmax' variable"

    # Uavg, VPeff (wind and vapour pressure)
    ds_uavg = ozwald_daily(
        query,
        variables=['Uavg', 'VPeff'],
        save_netcdf=True,
        save_json=False,
        plot=False,
        verbose=True
    )
    assert 'Uavg' in ds_uavg.data_vars, "Missing 'Uavg' variable"

    print("[done] ozwald_daily passed")
    return ds_pg, ds_tmax, ds_uavg


if __name__ == '__main__':
    test_ozwald_daily()
