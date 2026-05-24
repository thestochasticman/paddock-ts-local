# Paddock time series

Three closely-related functions that turn per-pixel rasters into
per-paddock summaries:

| Function | What it produces |
|---|---|
| `make_paddock_time_series` | Per-paddock medians for every band + index at every timestep, on dims `(paddock, time)`. |
| `make_yearly_paddock_time_series` | The same dataset split into one slice per calendar year, with a `doy` (day-of-year) coordinate attached. |
| `make_smoothed_paddock_time_series` | The same data resampled to a fixed cadence, gap-filled with PCHIP interpolation, and smoothed with a Savitzky-Golay filter. |

All three persist Zarr v2 outputs under `query.tmp_dir`, named from the
paddocks file's stem:

- `{stem}_timeseries.zarr`
- `{stem}_timeseries_{year}.zarr` (one per year)
- `{stem}_timeseries_smoothed.zarr`

These functions are the pivot from pixel-space to paddock-space — the
central time-series dataset that phenology and plotting consume.

---

## Example: produce a paddock × time table

```python
from datetime import date
from PaddockTS.query import Query
from PaddockTS.Phenology.make_paddock_time_series import make_paddock_time_series

q = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2024, 1, 1),
    end=date(2024, 12, 31),
    stub="ts_demo",
)

ts = make_paddock_time_series(q)
print(ts)
# <xarray.Dataset>
# Dimensions:      (paddock: 12, time: 73)
# Coordinates:
#   * paddock      (paddock) <U3 '1' '2' '3' ... '12'
#   * time         (time) datetime64[ns] 2024-01-03 ... 2024-12-29
#     spatial_ref  int32 ...
# Data variables:
#     nbart_blue   (paddock, time) float64 ...
#     nbart_green  (paddock, time) float64 ...
#     ...
#     NDVI         (paddock, time) float64 ...
#     CFI          (paddock, time) float64 ...
#     NIRv         (paddock, time) float64 ...
#     NDTI         (paddock, time) float64 ...
#     CAI          (paddock, time) float64 ...

# NDVI of paddock 1 over the year, as a pandas Series
ndvi_p1 = ts.NDVI.sel(paddock="1").to_pandas()
ndvi_p1.plot()
```

The `paddock` coordinate is always strings — `'1'`, `'2'`, …  — so
`ts.sel(paddock="1")` works whether your IDs are numeric or
human-readable labels.

---

## Example: split by year + plot DOY-aligned NDVI

`make_yearly_paddock_time_series` adds a `doy` (1–366) coordinate so
multi-year series can be overlaid on a common DOY axis:

```python
import matplotlib.pyplot as plt
from PaddockTS.Phenology.make_yearly_paddock_time_series import make_yearly_paddock_time_series

# Use a multi-year query
q = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2022, 1, 1),
    end=date(2024, 12, 31),
    stub="yearly_demo",
)
yearly = make_yearly_paddock_time_series(q)
# {2022: <Dataset>, 2023: <Dataset>, 2024: <Dataset>}

fig, ax = plt.subplots()
for year, ds in yearly.items():
    ax.plot(ds.doy, ds.NDVI.sel(paddock="1"), label=str(year))
ax.set_xlabel("DOY")
ax.set_ylabel("NDVI")
ax.legend()
```

---

## Example: smoothed series for phenology

Sentinel-2 revisit gaps and cloud-mask drops leave irregular series.
`make_smoothed_paddock_time_series` produces a uniform, smoothed
version suitable for phenology fitting and plotting:

```python
from PaddockTS.Phenology.make_smoothed_paddock_time_series import make_smoothed_paddock_time_series

smoothed = make_smoothed_paddock_time_series(
    q,
    days=10,          # 10-day median resample
    window_length=7,  # Savitzky-Golay window (odd; coerced if even)
    polyorder=2,      # SG polynomial order (< window_length)
)

# Compare raw vs. smoothed for one paddock
raw = ts.NDVI.sel(paddock="1")
smo = smoothed.NDVI.sel(paddock="1")
ax = raw.plot.scatter(x="time", label="raw")
smo.plot(ax=ax.axes, color="red", label="smoothed")
ax.axes.legend()
```

---

## Bring your own paddocks

All three functions accept `paddocks_filepath`. If `None`, they default
to the SAM paddocks at `query.sam_paddocks_path` (running SAM if it
hasn't been run yet). Any GeoPackage / Shapefile / GeoJSON with a
`paddock` column works; `load_user_paddocks` will derive missing
columns.

```python
ts = make_paddock_time_series(
    q,
    paddocks_filepath="/path/to/my_paddocks.gpkg",
)
```

The output Zarr is named from the file stem
(`my_paddocks_timeseries.zarr`), so SAM and user runs coexist in the
same `tmp_dir` without overwriting each other.

---

## Reference

### `make_paddock_time_series`

::: PaddockTS.Phenology.make_paddock_time_series.make_paddock_time_series

### `make_yearly_paddock_time_series`

::: PaddockTS.Phenology.make_yearly_paddock_time_series.make_yearly_paddock_time_series

### `make_smoothed_paddock_time_series`

::: PaddockTS.Phenology.make_smoothed_paddock_time_series.make_smoothed_paddock_time_series

### `split_paddock_time_series_by_year`

::: PaddockTS.Phenology.make_yearly_paddock_time_series.split_paddock_time_series_by_year
