# Spectral indices

Five vegetation / soil indices computed from Sentinel-2 surface
reflectance:

| Index | Formula | What it captures |
|---|---|---|
| **NDVI** | `(NIR − Red) / (NIR + Red)` | Greenness. Standard vegetation vigour proxy. |
| **CFI** | `NDVI × (Red + 2·Green − Blue)` | Crop Foliage Index — NDVI weighted by visible-band greenness. |
| **NIRv** | `NDVI × NIR` | Near-Infrared Reflectance of Vegetation — often a stronger GPP correlate than NDVI. |
| **NDTI** | `(SWIR2 − SWIR3) / (SWIR2 + SWIR3)` | Normalised Difference Tillage Index — sensitive to crop residue / lignin. |
| **CAI** | `0.5·(SWIR2 + SWIR3) − NIR` | Cellulose Absorption Index — distinguishes dry plant matter from bare soil. |

Each per-pixel array is float32, shape `(y, x, time)`. Zero DN values
are treated as Sentinel-2 ARD nodata (set to `NaN`) and DN values are
scaled by `1/10000` to reflectance in `[0, 1]` before any arithmetic.

The high-level entry point `compute_indices` adds every requested
index as a new `(time, y, x)` data variable to the input dataset and
persists the augmented cube to `query.indices_path` (Zarr v2,
guarded by a `_SUCCESS` marker).

---

## Example: full set in one call

```python
from datetime import date
from PaddockTS.query import Query
from PaddockTS.SpectralIndices.indices import compute_indices

q = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2024, 1, 1),
    end=date(2024, 3, 31),
    stub="indices_demo",
)

# `compute_indices` will download + clean Sentinel-2 on demand if needed
ds = compute_indices(q)

print(list(ds.data_vars))
# [..., 'NDVI', 'CFI', 'NIRv', 'NDTI', 'CAI']

# Median NDVI across the time window
ndvi_median = ds.NDVI.median(dim="time")
ndvi_median.plot.imshow(cmap="RdYlGn", vmin=-0.2, vmax=0.9)
```

---

## Example: a single index, no Zarr write

The five `compute_*` helpers return raw numpy arrays — useful when
you want a quick number without persisting an augmented cube:

```python
import xarray as xr
from PaddockTS.SpectralIndices.indices import compute_ndvi

ds = xr.open_zarr(q.sentinel2_clean_path, decode_coords="all")
ndvi = compute_ndvi(ds)         # numpy.ndarray, shape (y, x, time)
print(ndvi.shape, ndvi.dtype)   # (257, 197, 19) float32
```

---

## Example: custom index set

`compute_indices` takes an optional `indices={name: callable}` mapping.
Use it to compute a subset, swap in your own indices, or both:

```python
import numpy as np
from PaddockTS.SpectralIndices.indices import compute_ndvi, _band

def compute_evi(ds):
    """Enhanced Vegetation Index (Huete 2002)."""
    nir = _band(ds, "nbart_nir_1")
    red = _band(ds, "nbart_red")
    blue = _band(ds, "nbart_blue")
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)

ds = compute_indices(
    q,
    indices={
        "NDVI": compute_ndvi,
        "EVI": compute_evi,
    },
)
print(list(ds.data_vars))
# [..., 'NDVI', 'EVI']
```

Note: `_band` is a module-internal helper, not part of the public API.
For a stable contract, mirror the boilerplate yourself
(`np.float32`, zero → NaN, divide by 10 000).

---

## Reference

::: PaddockTS.SpectralIndices.indices
