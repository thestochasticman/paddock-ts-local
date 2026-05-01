#!/usr/bin/env python3
"""Test bbox conversion accuracy between lat/lon and bbox formats."""

from PaddockTS.query import Query
from datetime import date
import math


def bbox_to_lat_lon_buffer(bbox):
    """Extract center lat/lon and buffer from bbox.

    Args:
        bbox: [west, east, north, south]

    Returns:
        tuple: (lat, lon, buffer_km)
    """
    west, east, north, south = bbox

    # Calculate center
    lon = (west + east) / 2
    lat = (north + south) / 2

    # Calculate buffer in km from the differences
    lat_diff_degrees = (north - south) / 2
    lon_diff_degrees = (east - west) / 2

    # Convert to km
    lat_buffer_km = lat_diff_degrees * 111.0
    lon_buffer_km = lon_diff_degrees * 111.0 * math.cos(math.radians(lat))

    # Average the two (they should be close if conversion is accurate)
    buffer_km = (lat_buffer_km + lon_buffer_km) / 2

    return lat, lon, buffer_km, lat_buffer_km, lon_buffer_km


def test_roundtrip():
    """Test that lat/lon -> bbox -> lat/lon gives same results."""

    test_cases = [
        {"lat": -35.28, "lon": 149.11, "buffer_km": 10, "name": "Canberra, 10km"},
        {"lat": -33.52, "lon": 148.37, "buffer_km": 1, "name": "Near example query, 1km"},
        {"lat": 0.0, "lon": 0.0, "buffer_km": 5, "name": "Equator, 5km"},
        {"lat": 60.0, "lon": 10.0, "buffer_km": 10, "name": "High latitude, 10km"},
        {"lat": -60.0, "lon": -10.0, "buffer_km": 10, "name": "High south latitude, 10km"},
    ]

    print("=" * 80)
    print("ROUNDTRIP CONVERSION TEST: lat/lon/buffer_km -> bbox -> lat/lon/buffer_km")
    print("=" * 80)

    for tc in test_cases:
        print(f"\n{tc['name']}")
        print(f"  Original: lat={tc['lat']:.6f}, lon={tc['lon']:.6f}, buffer_km={tc['buffer_km']:.3f}")

        # Create query from lat/lon
        query = Query.from_lat_lon(
            lat=tc['lat'],
            lon=tc['lon'],
            buffer_km=tc['buffer_km'],
            start=date(2023, 1, 1),
            end=date(2023, 12, 31)
        )

        print(f"  Generated bbox: {[f'{x:.6f}' for x in query.bbox]}")

        # Convert back
        lat_back, lon_back, buffer_km_back, lat_buf, lon_buf = bbox_to_lat_lon_buffer(query.bbox)

        print(f"  Recovered: lat={lat_back:.6f}, lon={lon_back:.6f}, buffer_km={buffer_km_back:.3f}")
        print(f"    (lat_buffer={lat_buf:.3f} km, lon_buffer={lon_buf:.3f} km)")

        # Calculate errors
        lat_error = abs(tc['lat'] - lat_back)
        lon_error = abs(tc['lon'] - lon_back)
        buffer_error = abs(tc['buffer_km'] - buffer_km_back)

        print(f"  Errors: lat={lat_error:.9f}°, lon={lon_error:.9f}°, buffer={buffer_error:.6f} km")

        # Check if errors are within tolerance
        if lat_error < 1e-6 and lon_error < 1e-6 and buffer_error < 0.001:
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL")


def test_example_query():
    """Test with the example query from utils.py."""

    print("\n" + "=" * 80)
    print("EXAMPLE QUERY ANALYSIS")
    print("=" * 80)

    # Example bbox from utils.py
    bbox = [148.36265, -33.52606, 148.38265, -33.50606]
    print(f"\nExample bbox: {bbox}")
    print("  Format interpretation: [west, north, east, south]")

    lat, lon, buffer_km, lat_buf, lon_buf = bbox_to_lat_lon_buffer(bbox)

    print(f"\nExtracted parameters:")
    print(f"  Center: lat={lat:.6f}, lon={lon:.6f}")
    print(f"  Buffer: {buffer_km:.3f} km (avg)")
    print(f"    Latitude buffer: {lat_buf:.3f} km")
    print(f"    Longitude buffer: {lon_buf:.3f} km")
    print(f"  Difference: {abs(lat_buf - lon_buf):.3f} km")

    # Now create a new query with these parameters
    print(f"\nRecreating query with Query.from_lat_lon()...")
    query = Query.from_lat_lon(
        lat=lat,
        lon=lon,
        buffer_km=buffer_km,
        start=date(2020, 1, 1),
        end=date(2024, 12, 31)
    )

    print(f"  Original bbox: {bbox}")
    print(f"  Recreated bbox: {query.bbox}")
    print(f"  Differences: {[f'{abs(a-b):.9f}' for a, b in zip(bbox, query.bbox)]}")


if __name__ == "__main__":
    test_roundtrip()
    test_example_query()
