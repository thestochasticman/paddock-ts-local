"""Test ozwald_8day download."""
from PaddockTS.Data.Environmental import ozwald_8day
from tests.conftest import get_test_query, cleanup_test_data


def test_ozwald_8day():
    query = get_test_query()
    cleanup_test_data(query)

    print("\n=== Testing ozwald_8day ===")
    ds = ozwald_8day(
        query,
        variables=['Ssoil', 'Qtot', 'LAI', 'GPP'],
        save_netcdf=True,
        save_json=False,
        plot=False,
        verbose=True
    )
    assert ds is not None, "ozwald_8day returned None"
    assert 'LAI' in ds.data_vars, "Missing 'LAI' variable"
    assert 'GPP' in ds.data_vars, "Missing 'GPP' variable"
    assert 'Ssoil' in ds.data_vars, "Missing 'Ssoil' variable"
    assert 'Qtot' in ds.data_vars, "Missing 'Qtot' variable"
    print("[done] ozwald_8day passed")
    return ds


if __name__ == '__main__':
    test_ozwald_8day()
