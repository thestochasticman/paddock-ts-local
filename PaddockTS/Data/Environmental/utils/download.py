from xarray import open_dataset
from io import BytesIO
from os import environ
# from os import remove
# import aiofiles
import asyncio
# import httpx
import urllib3
import pyfive
import httpx

environ['PYTHON_GIL'] = '0'
environ["H5NETCDF_READ_BACKEND"] = "pyfive"  # thread-safe, pure Python

_http = urllib3.PoolManager()


# async def download_async(url: str, filename: str, chunk_size: int = 32768):
#     async with httpx.AsyncClient(timeout=30.0) as client:
#         async with client.stream('GET', url) as response:
#             if response.status_code == 200:
#                 async with aiofiles.open(filename, 'wb') as f:
#                     async for chunk in response.aiter_bytes(chunk_size=chunk_size):
#                         await f.write(chunk)


# def download_sync(url: str, filename: str, chunk_size=32768):
#     response = _http.request('GET', url, preload_content=False)
#     with open(filename, 'wb') as f:
#         for chunk in response.stream(chunk_size):
#             f.write(chunk)
#     response.release_conn()


def download(url: str) -> pyfive.File:
    response = httpx.get(url, timeout=None)
    response.raise_for_status()
    return pyfive.File(BytesIO(response.content))


def test():
    silo_baseurl = 'https://s3-ap-southeast-2.amazonaws.com/silo-open-data/Official/annual'
    url = f'{silo_baseurl}/daily_rain/2020.daily_rain.nc'
    filename = 'test.nc'
    download(url, filename)  # or: asyncio.run(download_async(url, filename))
    # asyncio.run(download_async(url, filename))
    print('hi')


if __name__ == '__main__':
    import time
    start = time.perf_counter()
    test()
    print(f'Elapsed: {time.perf_counter() - start:.2f}s')
