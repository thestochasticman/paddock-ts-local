"""Test daesim_forcing and daesim_soils."""
from PaddockTS.Data.Environmental import (
    ozwald_daily,
    ozwald_8day,
    silo_daily,
    daesim_forcing,
    daesim_soils,
)
from tests.conftest import get_test_query, cleanup_test_data


def test_daesim_forcing():
    query = get_test_query()
    cleanup_test_data(query)

    print("\n=== Downloading dependencies ===")
    ds_silo = silo_daily(
        query,
        variables=['radiation', 'vp', 'max_temp', 'min_temp', 'daily_rain', 'et_morton_actual', 'et_morton_potential'],
        save_netcdf=True,
        save_json=False,
        plot=False,
        verbose=True
    )
    ds_ozwald_8day = ozwald_8day(
        query,
        variables=['Ssoil', 'Qtot', 'LAI', 'GPP'],
        save_netcdf=True,
        save_json=False,
        plot=False,
        verbose=True
    )
    ds_pg = ozwald_daily(query, variables=['Pg'], save_netcdf=True, save_json=False, plot=False, verbose=True)
    ds_tmax = ozwald_daily(query, variables=['Tmax', 'Tmin'], save_netcdf=True, save_json=False, plot=False, verbose=True)
    ds_uavg = ozwald_daily(query, variables=['Uavg', 'VPeff'], save_netcdf=True, save_json=False, plot=False, verbose=True)

    print("\n=== Testing daesim_forcing ===")
    df = daesim_forcing(
        query,
        ds_silo_daily=ds_silo,
        ds_ozwald_8day=ds_ozwald_8day,
        ds_ozwald_daily_Pg=ds_pg,
        ds_ozwald_daily_Tmax=ds_tmax,
        ds_ozwald_daily_Uavg=ds_uavg,
        verbose=True
    )
    assert df is not None, "daesim_forcing returned None"
    assert 'Precipitation' in df.columns, "Missing 'Precipitation' column"
    assert 'SRAD' in df.columns, "Missing 'SRAD' column"
    print("[done] daesim_forcing passed")
    return df


def test_daesim_soils():
    query = get_test_query()
    cleanup_test_data(query)

    print("\n=== Testing daesim_soils ===")
    df = daesim_soils(query, verbose=True)
    assert df is not None, "daesim_soils returned None"
    assert 'Clay' in df.columns, "Missing 'Clay' column"
    assert 'Sand' in df.columns, "Missing 'Sand' column"
    print("[done] daesim_soils passed")
    return df


if __name__ == '__main__':
    test_daesim_forcing()
    test_daesim_soils()
