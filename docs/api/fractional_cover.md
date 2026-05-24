# Fractional cover

Per-pixel spectral unmixing of Sentinel-2 surface reflectance into
three ground-cover fractions:

| Band | Meaning |
|---|---|
| `bg`  | bare ground / soil |
| `pv`  | green (photosynthetic) vegetation |
| `npv` | non-green (non-photosynthetic) vegetation — senescent crop, stubble, woody material |

The model is a small TFLite MLP adapted from
[`fractionalcover3`](https://github.com/jrsrp/fractionalcover3) by
Robert Denham (MIT-licensed; see
`PaddockTS/LICENSES/fractionalcover3.LICENSE`). Four model variants
ship inside the package as bundled `.tflite` files at
`PaddockTS/FractionalCover/_models/`, indexed `n=1..4` from least to
most complex. `n=4` is the default and most accurate.

Output is per-pixel per-timestep, persisted to
`query.fractional_cover_path` as Zarr v2 (guarded by a `_SUCCESS`
marker).

---

## Example: compute and inspect

```python
from datetime import date
from PaddockTS.query import Query
from PaddockTS.FractionalCover import compute_fractional_cover

q = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2024, 1, 1),
    end=date(2024, 3, 31),
    stub="fc_demo",
)

# Will download + clean Sentinel-2 on demand if needed
fc = compute_fractional_cover(q)

print(fc)
# <xarray.Dataset>
# Dimensions:  (time: 19, y: 257, x: 197)
# Data variables:
#     bg   (time, y, x) float64 ...
#     pv   (time, y, x) float64 ...
#     npv  (time, y, x) float64 ...

# Mean green-vegetation fraction across the AOI over time
fc.pv.mean(dim=("y", "x")).plot()
```

---

## Example: render as a false-colour RGB

The fractional-cover video stage maps `(bg, pv, npv) → (R, G, B)` for
intuitive at-a-glance interpretation: red = bare, green = growing,
blue = stubble / dry. To reproduce that scheme on a single timestep:

```python
import matplotlib.pyplot as plt
import numpy as np

fc_t = fc.isel(time=0)
total = np.maximum(fc_t.bg + fc_t.pv + fc_t.npv, 1e-6)
rgb = np.stack([
    (fc_t.bg / total).values,
    (fc_t.pv / total).values,
    (fc_t.npv / total).values,
], axis=-1)
rgb = np.clip(np.nan_to_num(rgb), 0, 1)

plt.imshow(rgb)
plt.axis("off")
plt.title(f"Fractional cover — {str(fc.time.values[0])[:10]}")
```

---

## Choosing a model variant

```python
fc = compute_fractional_cover(q, model_n=2)  # smaller, faster, less accurate
```

Use `model_n=4` (default) unless you have a specific need to trade
accuracy for runtime; the model is small enough that the difference is
typically negligible for AOIs under a few thousand pixels per side.

---

## Calibration correction

The `correction=True` flag applies per-band sensor calibration
factors (gains and offsets fitted in the upstream `fractionalcover3`
work) instead of the simple `× 0.0001` DN-to-reflectance scaling:

```python
fc = compute_fractional_cover(q, correction=True)
```

Use this **only** when your inputs match the calibration assumptions
of the original model — i.e. raw Landsat-like DN. For ARD-corrected
DEA Sentinel-2 the default scaling is correct.

---

## Reference

::: PaddockTS.FractionalCover.compute_fractional_cover.compute_fractional_cover

### `BANDS`

The six Sentinel-2 SR bands stacked into the model input, in this
order:

```python
BANDS = [
    "nbart_blue",
    "nbart_green",
    "nbart_red",
    "nbart_nir_1",
    "nbart_swir_2",
    "nbart_swir_3",
]
```
