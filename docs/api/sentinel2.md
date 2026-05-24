# Sentinel-2 download

Two stages live here:

- `download_sentinel2` — fetches the **raw** Sentinel-2 ARD cube
  (including the fmask quality band) from a STAC catalog and writes it
  as Zarr v2 to `query.sentinel2_path`. The default STAC source is
  Geoscience Australia's [Digital Earth Australia](https://explorer.dea.ga.gov.au/)
  ARD collection (`ga_s2am_ard_3` / `ga_s2bm_ard_3`).
- `clean_sentinel2` — reads the raw cube, applies an fmask-based
  clear-sky pixel mask (drops the fmask band itself in the process),
  drops scenes whose NaN fraction exceeds `max_nan_fraction`, and
  writes the cleaned cube as Zarr v2 to `query.sentinel2_clean_path`.

Both writes are guarded by `_SUCCESS` markers — same query re-runs
reuse the cached file; different bbox or different time range gets its
own folder.

---

## What you get

The returned `xarray.Dataset` is keyed on `(time, y, x)` with the
following data variables (raw cube):

```text
nbart_blue        nbart_red_edge_1    nbart_swir_2
nbart_green       nbart_red_edge_2    nbart_swir_3
nbart_red         nbart_red_edge_3    oa_fmask        ← dropped by clean
                  nbart_nir_1
                  nbart_nir_2
```

Values are 16-bit DN (digital numbers, scale ~ `0–10000`). Convert to
reflectance with `* 0.0001`. The dataset carries a `spatial_ref`
coordinate so `ds.rio.crs` is populated after `xr.open_zarr(...,
decode_coords='all')`.

---

## Example: raw download

```python
from datetime import date
from PaddockTS.query import Query
from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2

q = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2024, 1, 1),
    end=date(2024, 3, 31),
    stub="s2_demo",
)

ds = download_sentinel2(q)
print(ds)
# <xarray.Dataset>
# Dimensions:      (time: 19, y: 257, x: 197)
# Coordinates:
#   * y            (y) ...
#   * x            (x) ...
#   * time         (time) datetime64[ns] 2024-01-03T00:23:36 ...
#     spatial_ref  int64 0
# Data variables:
#     nbart_blue   (time, y, x) uint16 ...
#     nbart_green  (time, y, x) uint16 ...
#     ...
#     oa_fmask     (time, y, x) uint8 ...
```

---

## Example: clean (mask clouds + drop bad scenes)

```python
from PaddockTS.Sentinel2.clean_sentinel2 import clean_sentinel2

ds_clean = clean_sentinel2(q, ds_sentinel2=ds, max_nan_fraction=0.5)

print("scenes before:", ds.time.size)
print("scenes after :", ds_clean.time.size)
print("fmask present:", "oa_fmask" in ds_clean.data_vars)  # False
```

`max_nan_fraction=0.5` keeps scenes that are at least ~50% clear
within the AOI. Lower to be more aggressive about discarding cloudy
scenes; raise to keep more time points for sparse acquisitions.

---

## Customising the catalog / bands

`download_sentinel2` accepts a `Sentinel2` config object (see
`PaddockTS.Sentinel2.sentinel2.Sentinel2`) controlling STAC URL,
collections, bands, CRS, resolution, and fmask values. The default —
`defaultsentinel2` — targets DEA ARD, 10 m, EPSG:6933, fmask cloud=2 /
shadow=3:

```python
from PaddockTS.Sentinel2.sentinel2 import Sentinel2

custom = Sentinel2(
    bands=('oa_fmask', 'nbart_red', 'nbart_green', 'nbart_blue', 'nbart_nir_1'),
    resolution=20,
)
ds = download_sentinel2(q, sentinel2=custom)
```

---

## Streaming write

`download_sentinel2` stages the dataset lazily via `odc.stac.load`
(returns a Dask-backed `xarray.Dataset`) and writes it to Zarr with
`ds.to_zarr(...)` while the Dask client is still alive. Chunks are
fetched, written to disk, and released — **the full cube is never
held in the driver process's memory**. Peak memory is roughly
`num_workers × threads_per_worker × chunk_size`, independent of AOI
size or year count.

If you have a 50 km² AOI over 3 years and ~150 scenes, the old
`client.compute(ds) → .result()` path would have peaked at the
whole-cube footprint (multi-GB). The streaming write peaks at a few
MB per chunk in flight.

After the write completes, the function returns
`xr.open_zarr(query.sentinel2_path, chunks=None, decode_coords='all')`
— a stable eager reference to the persisted store, not the lazy Dask
view (which would re-fetch from STAC the moment the client shut down).

## Thread-env restoration

Dask sets `OMP_NUM_THREADS=1` (and the MKL / OpenBLAS equivalents) on
the parent process when its cluster spins up. Without restoration
those values persist into later pipeline steps and would cripple the
PyTorch threading in SAM segmentation. `download_sentinel2` snapshots
the originals on entry and restores them in a `finally` block, so
even a failed download leaves the env exactly as it found it.

## Failure modes

DEA's STAC fronting load-balancer can return a 504 on cold first
request. `download_sentinel2` ships with a `urllib3.Retry` policy
covering 408/429/502/503/504 with exponential backoff. See
[`diagnostics.md`](https://github.com/thestochasticman/paddock-ts-local/blob/main/diagnostics.md)
for the full reproduction and the open `RasterioIOError('Unsupported
Authorization Type')` issue.

---

## Reference

### `download_sentinel2`

::: PaddockTS.Sentinel2.download_sentinel2.download_sentinel2

### `clean_sentinel2`

::: PaddockTS.Sentinel2.clean_sentinel2.clean_sentinel2
