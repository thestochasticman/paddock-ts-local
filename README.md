# PaddockTS

**Paddock-scale time-series analysis of Australian agricultural land,
end-to-end from a single bounding box.**

Built at the [Borevitz Lab, Australian National
University](https://borevitzlab.anu.edu.au/) for ecologists, agronomists,
and remote-sensing researchers who want a reproducible path from raw
Sentinel-2 imagery to per-paddock greenness, ground cover, and
phenology.

[![Docs](https://img.shields.io/badge/docs-thestochasticman.github.io-2ea44f)](https://thestochasticman.github.io/paddock-ts-local/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)

📚 **Documentation:** <https://thestochasticman.github.io/paddock-ts-local/>

---

## What it does

Give PaddockTS a bounding box and a date range. It returns:

- **Segmented paddock polygons** — automatic field-boundary detection
  via [Segment Anything](https://segment-anything.com/), driven through
  [`segment-geospatial`](https://samgeo.gishub.org/) (`samgeo`) by
  Qiusheng Wu, over an NDWI Fourier-feature presegmentation image.
  Produces a clean GeoPackage of per-paddock geometry, area (ha), and
  shape compactness.
- **Per-paddock time series** — for every Sentinel-2 acquisition in your
  window, the median reflectance and the median NDVI / CFI / NIRv /
  NDTI / CAI inside each paddock, written as a single Zarr cube on
  `(paddock, time)`.
- **Fractional cover** — pixel-level unmixing of Sentinel-2 surface
  reflectance into bare ground (`bg`), green vegetation (`pv`), and
  non-green vegetation (`npv`), via a TFLite MLP adapted from
  [`fractionalcover3`](https://github.com/jrsrp/fractionalcover3).
- **Phenology metrics** — start, peak, and end of season DOY plus
  amplitudes and integrals, computed per paddock per year through a
  vendored [`phenolopy`](https://github.com/lewistrotter/phenolopy).
- **Environmental context** — Copernicus 30 m DEM (with derived slope,
  aspect, flow accumulation, TWI),
  [OzWALD](https://www.wenfo.org/ozwald/) and
  [SILO](https://www.longpaddock.qld.gov.au/silo/) daily climate, and
  [SLGA](https://esoil.io/TERNLandscapes/Public/Pages/SLGA/index.html)
  90 m soil texture / properties, all clipped to the same AOI.
- **Plots and videos** — true-colour and false-colour MP4 timelines,
  per-paddock thumbnail calendars, phenology curves with SoS / PoS / EoS
  markers, climate diagnostic panels.


---

## Install

### Conda (recommended)

The easiest path: a single environment file installs the geospatial
native stack (GDAL, PROJ, GEOS), the ML stack (PyTorch, TensorFlow),
and PaddockTS itself.

```bash
git clone https://github.com/thestochasticman/paddock-ts-local.git
cd paddock-ts-local
conda env create -f paddock-ts-env.yml
conda activate paddockts
pip install -e .
```

### pip

If you already have GDAL, PROJ, GEOS, and (optionally) CUDA installed
system-wide:

```bash
pip install -e.
```

PaddockTS targets Python ≥ 3.11.

### Configure (optional)

Default output and cache directories are `~/Documents/PaddockTS-Outputs`
and `~/Downloads/PaddockTS-Tmp`. Override and add credentials by
creating `~/.config/PaddockTS.json`:

```json
{
  "out_dir": "/data/paddockts/outputs",
  "tmp_dir": "/data/paddockts/tmp",
  "email": "you@example.org",
  "tern_api_key": "<your-tern-key>"
}
```

**Credentials:**
- `email` is required only by the SILO climate stage
- `tern_api_key` is required only by the SLGA soils stage
- **Get your TERN API key:** Generate one at <https://account.tern.org.au/>

The Sentinel-2 → PaddockTS chain itself works without any credentials.

**Alternative: Pass config directly**

Instead of creating a config file, you can pass configuration directly
to the `Query` constructor:

```python
from PaddockTS.config import Config
from PaddockTS.query import Query
from datetime import date

custom_config = Config(
    out_dir="/data/paddockts/outputs",
    tmp_dir="/data/paddockts/tmp",
    email="you@example.org",
    tern_api_key="<your-tern-key>"
)

query = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2020, 1, 1),
    end=date(2021, 12, 31),
    stub="my_run",
    config=custom_config  # Pass config directly
)
```

---

## Quick example

```python
from datetime import date
from PaddockTS.query import Query
from PaddockTS.get_outputs import get_outputs

query = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],  # west, south, east, north
    start=date(2020, 1, 1),
    end=date(2021, 12, 31),
    stub="my_first_run",
)

get_outputs(query)
```

This kicks off both pipelines in parallel and renders a live dashboard.
Subsequent runs of the same `Query` skip every cached step.

Outputs land under `~/Documents/PaddockTS-Outputs/<stub>/`:

| File | What's in it |
|---|---|
| `<stub>_paddocks.gpkg` | Segmented paddock polygons + area_ha + compactness |
| `<stub>_paddockTS.zarr` | Per-paddock medians for every band and index, on `(paddock, time)` |
| `<stub>_paddockTS_<year>.zarr` | Yearly slices with a DOY coordinate |
| `<stub>_sentinel2.mp4` | True-colour Sentinel-2 timeline |
| `<stub>_fractional_cover.mp4` | Bare/green/non-green RGB timeline |
| `<stub>_calendar_<year>.png` | Per-paddock thumbnail calendar |
| `<stub>_phenology.png` | SoS/PoS/EoS curves per paddock per year |
| `<stub>_topography.png` | Elevation, slope, aspect, flow accumulation |

---

## Pipeline at a glance

| Sentinel-2 → PaddockTS (13 stages) | Environmental (7 stages) |
|---|---|
| Download Sentinel-2 | Download terrain (Copernicus DEM) |
| Compute spectral indices | Download OzWALD daily climate |
| Compute fractional cover | Download SILO climate |
| Sentinel-2 video | Download SLGA soils |
| Segment paddocks (SAM) | OzWALD plot |
| Sentinel-2 + paddocks video | SILO plot |
| Fractional cover video | Terrain plot |
| Fractional cover + paddocks video | |
| Make paddock time series (per-paddock medians) | |
| Make yearly paddock time series | |
| Estimate phenology | |
| Calendar plot | |
| Phenology plot | |

Every stage is a standalone function — pick any subset, swap in your
own segmentation, plug in your own phenology library. See the
[pipeline page](https://thestochasticman.github.io/paddock-ts-local/pipeline/)
for the full call-graph and per-stage caching behaviour.

---

## Calling individual stages

```python
from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
from PaddockTS.SpectralIndices.indices import compute_indices
from PaddockTS.FractionalCover import compute_fractional_cover
from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks

ds = download_sentinel2(query)              # Zarr cube on disk
ds = compute_indices(query, ds_sentinel2=ds)  # NDVI, CFI, NIRv, NDTI, CAI
fc = compute_fractional_cover(query, ds_sentinel2=ds)  # bg / pv / npv
paddocks = get_paddocks(query, ds_sentinel2=ds)        # GeoDataFrame
```

Every function will load its own inputs from the cache if you don't
pass them, so you can also call them out of order or in isolation.

---

## Data sources and acknowledgments

PaddockTS does not redistribute upstream data; it queries them on demand:

- **Sentinel-2 ARD** — Geoscience Australia
  [Digital Earth Australia](https://explorer.dea.ga.gov.au/) STAC.
- **Copernicus DEM 30 m** —
  [AWS Open Data](https://registry.opendata.aws/copernicus-dem/).
- **OzWALD** — Australian Water and Landscape Dynamics, hosted by ANU.
- **SILO** — Queensland Government's gridded climate dataset.
- **SLGA** — TERN / CSIRO Soil and Landscape Grid of Australia
  (TERN API key required).

If you publish work that uses PaddockTS, please cite the upstream data
sources, the third-party libraries listed below, and the PaddockTS
repository.

## License and attribution

PaddockTS is **MIT-licensed** — see [LICENSE](LICENSE).

### Vendored code

Third-party code shipped inside the package; see
[`PaddockTS/LICENSES/`](PaddockTS/LICENSES/) for full license texts:

- [`fractionalcover3`](https://gitlab.com/jrsrp/themes/cover/fractionalcover3) by
  Robert Denham — MIT. The TFLite unmixing models and the unmixing
  routine in `PaddockTS.FractionalCover._unmix` are adapted from
  this work.
- [`phenolopy`](https://github.com/lewistrotter/phenolopy) by
  Lewis Trotter — Apache 2.0. Vendored verbatim under
  `PaddockTS.Phenology._phenolopy` (with minor NumPy 2.0 compatibility
  fixes documented in the file header) and used through
  `PaddockTS.Phenology.estimate_phenology`.
- [`DAESIM_preprocess`](https://github.com/ChristopherBradley/DAESIM_preprocess) by
  Christopher Bradley — MIT. Environmental data harvesting functions
  adapted in `PaddockTS.Environmental` for downloading
  and processing climate, vegetation, soil, and topographic datasets.

### Key runtime dependencies

Installed as regular dependencies, not vendored — please cite if
relevant to your work:

- [`segment-geospatial`](https://samgeo.gishub.org/) (`samgeo`) by
  Qiusheng Wu — MIT. Wraps Segment Anything for geospatial use; drives
  the paddock segmentation stage. Cite:
  [Wu & Osco (2023), J. Open Source Software](https://joss.theoj.org/papers/10.21105/joss.05663).
- [Segment Anything Model](https://segment-anything.com/) (SAM) by
  Meta AI Research — Apache 2.0. The underlying segmentation model.

## Contributing & support

- **Bug reports / feature requests:**
  [GitHub Issues](https://github.com/thestochasticman/paddock-ts-local/issues)
- **Documentation:**
  <https://thestochasticman.github.io/paddock-ts-local/>
- **Maintainers:** Borevitz Lab, Australian National University
