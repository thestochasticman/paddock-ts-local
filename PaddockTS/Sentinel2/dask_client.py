from dask.distributed import Client as DaskClient
from typing import TYPE_CHECKING, Generator
from contextlib import contextmanager

@contextmanager
def dask_client(
    num_workers: int = 4,
    threads_per_worker: int = 2,
) -> Generator[DaskClient, None, None]:
    
    client = DaskClient(
        n_workers=num_workers,
        threads_per_worker=threads_per_worker,
    )
    try:
        yield client
    except Exception as e:
        print(f"Dask computation failed: {e}")
        raise
    finally:
        client.close()