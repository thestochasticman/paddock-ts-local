import asyncio
import time
import xarray as xr
import httpx
from io import BytesIO

async def fetch_nc(url: str, task_id: int):
    print(f"[{task_id}] Starting download...")
    t0 = time.perf_counter()
    
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.get(url)
    
    t1 = time.perf_counter()
    print(f"[{task_id}] Download: {t1 - t0:.2f}s")
    
    ds = xr.open_dataset(BytesIO(response.content))
    
    t2 = time.perf_counter()
    print(f"[{task_id}] Parse: {t2 - t1:.3f}s")  # This is the GIL hold
    
    return ds


async def main():
    url = 'https://s3-ap-southeast-2.amazonaws.com/silo-open-data/Official/annual/daily_rain/2020.daily_rain.nc'
    
    # Run 5 concurrent requests
    t0 = time.perf_counter()
    tasks = [fetch_nc(url, i) for i in range(5)]
    await asyncio.gather(*tasks)
    
    print(f"\nTotal: {time.perf_counter() - t0:.2f}s")


if __name__ == '__main__':
    asyncio.run(main())