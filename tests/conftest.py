"""Shared test fixtures."""
import sys
from pathlib import Path

# Add project root to path for pytest
sys.path.insert(0, str(Path(__file__).parent.parent))

import shutil
from datetime import date
from os.path import exists
from PaddockTS.query import Query


def get_test_query() -> Query:
    """Create a small test query."""
    return Query(
        stub='test_environmental',
        lat=-33.5040,
        lon=148.4,
        buffer=0.005,
        start_time=date(2020, 1, 1),
        end_time=date(2020, 3, 31),
    )


def cleanup_test_data(query: Query):
    """Delete preexisting test data."""
    print("\n=== Cleaning up test data ===")
    for path in [query.stub_tmp_dir, query.stub_out_dir]:
        if exists(path):
            print(f"Removing {path}")
            shutil.rmtree(path)
    print("[done] Cleanup complete")
