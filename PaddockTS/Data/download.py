"""
Download all data (Sentinel-2 and Environmental) in parallel.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed

from PaddockTS.query import Query


def _run_sentinel2(query: Query, verbose: bool):
    from PaddockTS.Data.download_sentinel2 import download_sentinel2
    download_sentinel2(query)
    return 'sentinel2'


def _run_environmental(query: Query, verbose: bool):
    from PaddockTS.Data.download_environmental import download_environmental
    download_environmental(query, verbose=verbose)
    return 'environmental'


def download_all(query: Query, verbose=True):
    """Download all data for a query, running Sentinel-2 and Environmental in parallel.

    Parameters
    ----------
        query: Query object specifying location and time range
        verbose: Print progress messages

    Returns
    -------
        dict with results from both downloads
    """
    import os
    os.makedirs(query.stub_out_dir, exist_ok=True)
    os.makedirs(query.stub_tmp_dir, exist_ok=True)

    download_tasks = [
        (_run_sentinel2, 'sentinel2'),
        (_run_environmental, 'environmental'),
    ]

    if verbose:
        print(f"Starting parallel downloads: Sentinel-2 + Environmental...")

    results = {}
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(func, query, verbose): name for func, name in download_tasks}

        for future in as_completed(futures):
            task_name = futures[future]
            try:
                result = future.result()
                results[result] = True
                if verbose:
                    print(f"[done] Completed {task_name}")
            except Exception as e:
                if verbose:
                    print(f"[failed] Failed {task_name}: {e}")
                raise

    if verbose:
        print("\n[done] All downloads complete")

    return results
