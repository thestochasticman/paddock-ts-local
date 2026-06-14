# PaddockTS

**Paddock-scale time-series analysis of Australian agricultural land,
end-to-end from a single bounding box.**

PaddockTS turns a bounding box and a date range into a complete set of
paddock-scale geospatial outputs: segmented field polygons, vegetation
indices, fractional ground-cover, per-paddock time series, seasonal
phenology metrics, and review-grade plots and videos — alongside
co-registered terrain, climate, and soil context.

Built at the [Borevitz Lab, Australian National
University](https://borevitzlab.anu.edu.au/) for ecologists, agronomists,
and remote-sensing researchers who want a reproducible path from raw
Sentinel-2 imagery to per-paddock greenness, ground cover, and
phenology.

---

## What you get

Given a `Query` (a bounding box + date range), PaddockTS produces:

| Output | What it is |
|---|---|
| **Paddock polygons** | Automatic field-boundary detection via [Segment Anything](https://segment-anything.com/) on an NDWI Fourier presegmentation image. A clean GeoPackage with per-paddock geometry, `area_ha`, and `compactness`. |
| **Per-paddock time series** | For every Sentinel-2 acquisition, the median reflectance and the median NDVI / CFI / NIRv / NDTI / CAI inside each paddock, persisted as a Zarr cube on `(paddock, time)`. |
| **Fractional cover** | Per-pixel unmixing of S2 reflectance into bare ground (`bg`), green vegetation (`pv`), and non-green vegetation (`npv`) via a TFLite MLP adapted from [`fractionalcover3`](https://github.com/jrsrp/fractionalcover3). |
| **Phenology metrics** | Per paddock per year: start/peak/end of season DOY, amplitudes, integrals — via a vendored [`phenolopy`](https://github.com/lewistrotter/phenolopy). |
| **Environmental context** | Copernicus 30 m DEM (with derived slope, aspect, flow accumulation, TWI), [OzWALD](https://www.wenfo.org/ozwald/) and [SILO](https://www.longpaddock.qld.gov.au/silo/) daily climate, and [SLGA](https://esoil.io/TERNLandscapes/Public/Pages/SLGA/index.html) 90 m soil texture / properties, all clipped to the same AOI. |
| **Plots & videos** | True-colour and false-colour MP4 timelines, per-paddock thumbnail calendars, phenology curves with SoS / PoS / EoS markers, climate diagnostic panels, and a stitched PDF report. |

---

## Quick example

```python
from datetime import date
from PaddockTS.query import Query
from PaddockTS.get_outputs import get_outputs

query = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2020, 1, 1),
    end=date(2021, 12, 31),
    stub="my_first_run",
)

get_outputs(query)
```

This kicks off both pipelines (Sentinel-2 → PaddockTS and Environmental)
in parallel and renders a live two-column status dashboard. Outputs
land under `~/Documents/PaddockTS-Outputs/<stub>/` (configurable). The
next `get_outputs(query)` for the same `Query` is a no-op — every
stage finds its cached output and skips.

---

## Bring your own paddocks

If you already have paddock boundaries from QGIS, a cadastral layer, or
a previous run, skip SAM segmentation and use them directly:

```python
from datetime import date
from PaddockTS.query import Query
from PaddockTS.get_outputs import get_outputs

paddocks_fp = "/path/to/my_paddocks.gpkg"  # or .geojson / .shp

query = Query.build_from_paddocks(
    paddocks_filepath=paddocks_fp,
    start=date(2024, 1, 1),
    end=date(2024, 12, 31),
    stub="my_farm",
    label_col="paddock_name",  # column holding human-readable names
)

get_outputs(
    query,
    paddocks_filepath=paddocks_fp,
    skip_sam=True,
    label_col="paddock_name",
)
```

---

## Where to go next

- **[Getting started](getting-started.md)** — install, configure,
  construct a `Query`, and run your first pipeline.
- **[Pipeline](pipeline.md)** — every stage, what it produces, what it
  reads, what it caches, and how to skip or replace any of it.
- **[API reference](api/index.md)** — full signatures and runnable
  examples for every public function.
- **[Demo notebooks](https://github.com/thestochasticman/paddock-ts-local/tree/main/demo)** —
  three runnable Jupyter notebooks: the quickstart, calling stages
  individually, and using your own paddock boundaries.

---

## License

PaddockTS is **MIT-licensed** — see [LICENSE](https://github.com/thestochasticman/paddock-ts-local/blob/main/LICENSE).

It vendors third-party code under permissive licenses (see
[`PaddockTS/LICENSES/`](https://github.com/thestochasticman/paddock-ts-local/tree/main/PaddockTS/LICENSES)):

- [`fractionalcover3`](https://github.com/jrsrp/fractionalcover3) — Robert Denham, MIT
- [`phenolopy`](https://github.com/lewistrotter/phenolopy) — Lewis Trotter, Apache 2.0
- [`DAESIM_preprocess`](https://github.com/ChristopherBradley/DAESIM_preprocess) — Christopher Bradley, MIT

If you publish work using PaddockTS, please cite the upstream data
sources (DEA Sentinel-2 ARD, Copernicus DEM, OzWALD, SILO, SLGA) and
the third-party libraries above.
