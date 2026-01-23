"""Test download with cached SILO files (no cleanup)."""
from datetime import date
from PaddockTS.query import Query
from PaddockTS.Data.download_environmental import download_environmental
import shutil
import os


def test_download_cached():
    """Test download with SILO cache intact to measure speedup."""
    query = Query(
        stub='test_cached',
        lat=-33.5040,
        lon=148.4,
        buffer=0.005,
        start_time=date(2020, 1, 1),
        end_time=date(2021, 3, 31),
    )

    # Only clean up stub-specific dirs, NOT the shared SILO cache
    for path in [query.stub_tmp_dir, query.stub_out_dir]:
        if os.path.exists(path):
            print(f"Removing {path}")
            shutil.rmtree(path)

    # Check if SILO cache exists
    silo_folder = os.path.join(query.tmp_dir, "SILO")
    if os.path.exists(silo_folder):
        cached_files = os.listdir(silo_folder)
        print(f"\nSILO cache exists with {len(cached_files)} files")
    else:
        print("\nNo SILO cache - files will be downloaded")

    print(f"\n=== Testing download with cache ===")
    print(f"Query: {query.start_time} to {query.end_time}\n")

    import time
    start = time.time()
    results = download_environmental(query, verbose=True)
    elapsed = time.time() - start

    print(f"\n=== Results (completed in {elapsed:.1f}s) ===")
    for key, val in results.items():
        if hasattr(val, 'dims'):
            print(f"{key}: dims={dict(val.sizes)}")

    assert 'silo_daily' in results
    assert 'ozwald_8day' in results
    print("\n[done] Cached download test passed")


if __name__ == '__main__':
    print("=" * 50)
    print("Testing download with SILO cache...")
    print("=" * 50)

    test_download_cached()

    print("\n" + "=" * 50)
    print("Cached download test passed!")
    print("=" * 50)
