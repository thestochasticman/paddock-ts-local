"""Test Query class."""
import sys
from datetime import date
from PaddockTS.query import Query, get_example_query


def test_query_creation():
    """Test basic Query creation."""
    print("\n=== Testing Query creation ===")
    q = Query(
        lat=-33.5040,
        lon=148.4,
        buffer=0.01,
        start_time=date(2020, 1, 1),
        end_time=date(2020, 6, 1),
    )
    assert q.lat == -33.5040
    assert q.lon == 148.4
    assert q.buffer == 0.01
    assert q.start_time == date(2020, 1, 1)
    assert q.end_time == date(2020, 6, 1)
    print("[done] Query creation passed")


def test_query_properties():
    """Test computed properties."""
    print("\n=== Testing Query properties ===")
    q = Query(
        lat=-33.5040,
        lon=148.4,
        buffer=0.01,
        start_time=date(2020, 1, 1),
        end_time=date(2020, 6, 1),
    )

    assert q.centre == (q.lon, q.lat)
    assert q.lon_range == (q.lon - q.buffer, q.lon + q.buffer)
    assert q.lat_range == (q.lat - q.buffer, q.lat + q.buffer)
    assert q.datetime == "2020-01-01/2020-06-01"
    assert q.bbox == [q.lon_range[0], q.lat_range[0], q.lon_range[1], q.lat_range[1]]
    print("[done] Query properties passed")


def test_query_stub_generation():
    """Test automatic stub generation."""
    print("\n=== Testing Query stub generation ===")
    q1 = Query(
        lat=-33.5040,
        lon=148.4,
        buffer=0.01,
        start_time=date(2020, 1, 1),
        end_time=date(2020, 6, 1),
    )

    # Same parameters should generate same stub
    q2 = Query(
        lat=-33.5040,
        lon=148.4,
        buffer=0.01,
        start_time=date(2020, 1, 1),
        end_time=date(2020, 6, 1),
    )

    # Different parameters should generate different stub
    q3 = Query(
        lat=-33.5040,
        lon=148.5,  # different lon
        buffer=0.01,
        start_time=date(2020, 1, 1),
        end_time=date(2020, 6, 1),
    )

    assert q1.stub == q2.stub, "Same parameters should produce same stub"
    assert q1.stub != q3.stub, "Different parameters should produce different stub"
    assert len(q1.stub) == 64, "Stub should be 64 characters (SHA-256 hex)"
    print("[done] Query stub generation passed")


def test_query_custom_stub():
    """Test custom stub override."""
    print("\n=== Testing Query custom stub ===")
    q = Query(
        stub='my_custom_stub',
        lat=-33.5040,
        lon=148.4,
        buffer=0.01,
        start_time=date(2020, 1, 1),
        end_time=date(2020, 6, 1),
    )
    assert q.stub == 'my_custom_stub'
    print("[done] Query custom stub passed")


def test_query_date_converter():
    """Test string to date conversion."""
    print("\n=== Testing Query date converter ===")
    q = Query(
        lat=-33.5040,
        lon=148.4,
        buffer=0.01,
        start_time='2020-01-01',
        end_time='2020-06-01',
    )
    assert q.start_time == date(2020, 1, 1)
    assert q.end_time == date(2020, 6, 1)
    print("[done] Query date converter passed")


def test_query_from_cli():
    """Test CLI parsing."""
    print("\n=== Testing Query from_cli ===")
    original_argv = sys.argv.copy()
    sys.argv = [
        "prog",
        "--stub", "test_example_query",
        "--lat", "-33.5040",
        "--lon", "148.4",
        "--buffer", "0.01",
        "--start_time", "2020-01-01",
        "--end_time", "2020-06-01",
        "--collections", "ga_s2am_ard_3", "ga_s2bm_ard_3",
        "--bands", "nbart_blue", "nbart_green", "nbart_red",
        "--filter", "eo:cloud_cover < 10"
    ]

    try:
        q = Query.from_cli()
    finally:
        sys.argv = original_argv

    assert isinstance(q, Query)
    assert q.lat == -33.5040
    assert q.lon == 148.4
    assert q.buffer == 0.01
    assert q.start_time == date(2020, 1, 1)
    assert q.end_time == date(2020, 6, 1)
    assert q.collections == ["ga_s2am_ard_3", "ga_s2bm_ard_3"]
    assert q.bands == ["nbart_blue", "nbart_green", "nbart_red"]
    print("[done] Query from_cli passed")


def test_get_example_query():
    """Test example query helper."""
    print("\n=== Testing get_example_query ===")
    q = get_example_query()
    assert isinstance(q, Query)
    assert q.stub == 'test_example_query'
    print("[done] get_example_query passed")


def test_all():
    """Run all Query tests."""
    print("=" * 50)
    print("Running Query tests...")
    print("=" * 50)

    test_query_creation()
    test_query_properties()
    test_query_stub_generation()
    test_query_custom_stub()
    test_query_date_converter()
    test_query_from_cli()
    test_get_example_query()

    print("\n" + "=" * 50)
    print("All Query tests passed!")
    print("=" * 50)


if __name__ == '__main__':
    test_all()
