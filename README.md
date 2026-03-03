# PaddockTS

**End-to-end geospatial pipeline for agricultural paddock time-series analysis.**

PaddockTS ingests Sentinel-2 satellite imagery over a user-defined bounding box and date range, computes spectral vegetation indices and fractional cover, segments the landscape into individual paddock polygons, extracts per-paddock time series, estimates crop/pasture phenology, and produces video, calendar, and phenology visualisations — all bundled into a single PDF report.


---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Directory Structure](#directory-structure)
- [Pipeline Steps](#pipeline-steps)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Module Reference](#module-reference)
  - [Query & Config](#query--config)
  - [Sentinel-2 Download](#sentinel-2-download)
  - [Indices & Fractional Cover](#indices--fractional-cover)
  - [Paddock Segmentation](#paddock-segmentation)
  - [PaddockTS Time Series](#paddockts-time-series)
  - [Phenology](#phenology)
  - [Plotting & Reporting](#plotting--reporting)
  - [Environmental Data](#environmental-data)
- [Data Flow](#data-flow)
- [Output Artifacts](#output-artifacts)
- [Running Individual Modules](#running-individual-modules)
- [License](#license)

---

## Architecture Overview

```
              ┌─────────────────────────┐
              │    Query(bbox, dates)   │
              └────────────┬────────────┘
                           │
              ┌────────────┼──────────────────────────────────────┐
              │                                                   │
              ▼                                                   ▼
  ┌────────────────────┐          ┌────────────────────────────────────────────┐
  │ Sentinel-2 Download│          │          Environmental (standalone)        │
  └─────────┬──────────┘          │   SILO · OzWALD · Terrain · SLGA Soils     │
      ┌─────┴──────┐              └────────────────────────────────────────────┘
      ▼            ▼       
  ┌──────────┐ ┌─────────┐ 
  │ Indices  │ │ VegFrac │   NDVI/CFI/NIRv/NDTI/CAI │ bg/pv/npv
  └────┬─────┘ └────┬────┘ 
       │            │     
  ┌────▼────────────▼─────▼┐
  │  Paddock Segmentation  │  SAMGeo │ K-Means │ W-Net
  └────────────┬───────────┘
               │
  ┌────────────▼────────────┐
  │   PaddockTS Extraction  │  per-paddock median time series
  └────────────┬────────────┘
         ┌─────┴──────┐
         ▼            ▼
  ┌──────────┐   ┌───────────┐
  │ Phenology│   │  Plotting │  SoS/PoS/EoS │ videos, calendars, PDF
  └──────────┘   └───────────┘
```

---

## Directory Structure

```
paddock-ts-local/
├── pyproject.toml                          # Package metadata, build config, dependencies
├── paddock-ts-env.yml                      # Conda environment specification
├── env.yml                                 # Experimental free-threaded Python env
├── README.md                               # This file
├── .gitmodules                             # Git submodule references (phenolopy)
│
└── PaddockTS/                              # Main package
    ├── __init__.py
    ├── config.py                           # Global paths (out_dir, tmp_dir), loads ~/.config/PaddockTS.json
    ├── query.py                            # Query dataclass: bbox + dates → deterministic stub + paths
    ├── utils.py                            # get_example_query() helper
    ├── status.py                           # Check which outputs exist for a query
    ├── sentinel2_to_paddockTS_pipeline.py  # Orchestrator — runs all 13 steps with Rich progress table
    ├── run_environmental.py                # Environmental-only pipeline (terrain, OzWALD, SILO, SLGA + plots)
    ├── get_outputs.py                      # Concurrent orchestrator — runs S2 pipeline + environmental in parallel threads
    │
    ├── Sentinel2/                          # Step 1: Satellite imagery acquisition
    │   ├── __init__.py
    │   ├── sentinel2.py                    # Sentinel2 config (STAC URL, bands, CRS, resolution)
    │   └── download_sentinel2.py           # STAC search → Dask load → cloud mask → zarr
    │
    ├── IndicesAndVegFrac/                  # Steps 2–3: Spectral products
    │   ├── __init__.py
    │   ├── indices.py                      # NDVI, CFI, NIRv, NDTI, CAI computation
    │   └── veg_frac.py                     # Fractional cover via fractionalcover3 (bg, pv, npv)
    │
    ├── PaddockSegmentation/                # Step 5 (active): SAMGeo-based segmentation
    │   ├── _presegment.py                  # NDWI Fourier → 3-band uint8 preseg GeoTIFF
    │   └── get_paddocks.py                 # SAMGeo ViT-H → vectorise → filter by area/compactness
    │
    ├── PaddockSegmentation2/               # Alternative: K-Means + contour segmentation
    │   ├── preprocess.py                   # K-Means clustering, auto-k via scoring, edge detection
    │   ├── utils.py                        # NDVI/NDWI, timeseries K-Means, labels→paddocks, evaluation
    │   └── get_paddocks.py                 # Full pipeline: preseg → cluster → filter → .gpkg
    │
    ├── PaddockSegmentation3/               # Experimental: W-Net (PyTorch) unsupervised segmentation
    │   ├── wnet.py                         # Dual-UNet with Normalised Cut loss
    │   ├── utils.py                        # Shared helper functions
    │   └── get_paddocks.py                 # Train W-Net → segment → filter → .gpkg
    │
    ├── PaddockTS/                          # Steps 8–9: Per-paddock time-series extraction
    │   ├── __init__.py
    │   ├── make_paddockTS.py               # Rasterise paddocks, compute median per paddock per timestep
    │   ├── make_smoothed_paddockTS.py      # Resample → PCHIP interpolation → Savitzky-Golay smoothing
    │   └── make_yearly_paddockTS.py        # Split by year, add day-of-year coordinate
    │
    ├── Phenology/                          # Step 10: Phenological metric estimation
    │   ├── __init__.py
    │   ├── estimate_phenology.py           # SoS, PoS, EoS via phenolopy per paddock per year
    │   └── phenolopy/                      # Git submodule — phenolopy library
    │       ├── scripts/phenolopy.py        # Core phenolopy algorithm
    │       └── data/                       # Example MODIS NDVI rasters
    │
    ├── Plotting/                           # Steps 4, 6–7, 11–12 + reporting
    │   ├── __init__.py
    │   ├── sentinel2_video.py              # RGB composite → H.264 MP4 with date overlay
    │   ├── vegfrac_video.py                # bg/pv/npv → RGB video
    │   ├── sentinel2_paddocks_video.py     # Sentinel-2 video + rasterised paddock boundaries
    │   ├── vegfrac_paddocks_video.py       # VegFrac video + paddock boundaries
    │   ├── calendar_plot.py                # Per-year paddock thumbnail calendar grids (PIL)
    │   ├── phenology_plot.py               # Scatter + interpolated NDVI with SoS/PoS/EoS markers
    │   ├── silo_plot.py                    # SILO climate variable plots
    │   ├── ozwald_plot.py                  # OzWALD daily & 8-day plots
    │   ├── terrain_tiles_plot.py           # Elevation, slope, aspect, flow accumulation (pysheds)
    │   └── make_pdf.py                     # Assemble all plots into a single PDF report
    │
    └── Environmental/                      # Standalone environmental data acquisition
        ├── __init__.py
        ├── SILO/                           # Australian Bureau of Meteorology point climate
        │   ├── __init__.py
        │   ├── silo.py                     # 18 variables config (rainfall, temp, radiation, etc.)
        │   └── download_silo.py            # HTTP API → daily CSV
        │
        ├── OzWALD/                         # CSIRO satellite-derived land/water products
        │   ├── __init__.py
        │   ├── ozwald.py                   # Variable config: 9 daily meteo + 13 eight-day products
        │   ├── download_ozwald_daily.py    # THREDDS OPeNDAP → daily CSV
        │   └── download_ozwald_8day.py     # THREDDS OPeNDAP → 8-day CSV
        │
        ├── TerrainTiles/                   # Copernicus 30m DEM
        │   ├── __init__.py
        │   ├── download_terrain_tiles.py   # S3 COG download → merged GeoTIFF
        │   ├── download_cog.py             # Generic COG downloader with bbox windowing
        │   └── utils.py                    # Flow accumulation, slope, TWI (pysheds)
        │
        └── SLGASoils/                      # TERN Soil & Landscape Grid of Australia
            ├── __init__.py
            ├── slgasoils.py                # 11 soil attributes × 6 depth ranges config
            ├── download_slgasoils.py       # TERN API → COG → GeoTIFF per (attribute, depth)
            ├── download_cog.py             # Authenticated COG download via GDAL /vsicurl/
            ├── utils.py                    # TERN API key loading, auth setup
            └── plot.py                     # Per-attribute raster visualisation
```

---

## Pipeline Steps

The main orchestrator (`sentinel2_to_paddockTS_pipeline.py`) executes 13 steps sequentially, displaying a Rich live-updating progress table:

| #  | Step                        | Module                                   | Output                         |
|----|-----------------------------|------------------------------------------|--------------------------------|
| 1  | Download Sentinel-2         | `Sentinel2.download_sentinel2`           | `{stub}_sentinel2.zarr`       |
| 2  | Compute indices             | `IndicesAndVegFrac.indices`              | In-memory (NDVI, CFI, etc.)    |
| 3  | Compute vegfrac             | `IndicesAndVegFrac.veg_frac`             | `{stub}_vegfrac.zarr`         |
| 4  | Sentinel-2 video            | `Plotting.sentinel2_video`               | `{stub}_sentinel2.mp4`        |
| 5  | Segment paddocks            | `PaddockSegmentation.get_paddocks`       | `{stub}_paddocks.gpkg`        |
| 6  | Sentinel-2 + paddocks video | `Plotting.sentinel2_paddocks_video`      | `{stub}_sentinel2_paddocks.mp4` |
| 7  | Vegfrac video               | `Plotting.vegfrac_video`                 | `{stub}_vegfrac.mp4`          |
| 8  | Vegfrac + paddocks video    | `Plotting.vegfrac_paddocks_video`        | `{stub}_vegfrac_paddocks.mp4` |
| 9  | Make paddockTS              | `PaddockTS.make_paddockTS`               | `{stub}_paddockTS.zarr`       |
| 10 | Make yearly paddockTS       | `PaddockTS.make_yearly_paddockTS`        | `{stub}_paddockTS_{year}.zarr` |
| 11 | Estimate phenology          | `Phenology.estimate_phenology`           | In-memory (SoS, PoS, EoS)     |
| 12 | Calendar plot               | `Plotting.calendar_plot`                 | `{stub}_calendar_{year}.png`  |
| 13 | Phenology plot              | `Plotting.phenology_plot`                | `{stub}_phenology.png`        |

Each step caches its output. Re-runs skip steps whose artifacts already exist on disk. Pass `reload=True` to clear cached intermediates and rerun from scratch.

### Environmental Pipeline (`run_environmental.py`)

A separate orchestrator for environmental data only, with 7 steps:

| # | Step              | Module                                        | Output                    |
|---|-------------------|-----------------------------------------------|---------------------------|
| 1 | Download terrain  | `Environmental.TerrainTiles.download_terrain`  | `{stub}_terrain.tif`      |
| 2 | Download OzWALD   | `Environmental.OzWALD.download_ozwald_daily`   | `{stub}_ozwald_daily.csv` |
| 3 | Download SILO     | `Environmental.SILO.download_silo`             | `{stub}_silo.csv`         |
| 4 | Download SLGA     | `Environmental.SLGASoils.download_slga_soils`  | Soil GeoTIFFs             |
| 5 | OzWALD plot       | `Plotting.ozwald_plot`                         | `{stub}_ozwald_*.png`     |
| 6 | SILO plot         | `Plotting.silo_plot`                           | `{stub}_silo_*.png`       |
| 7 | Terrain plot      | `Plotting.terrain_tiles_plot`                  | `{stub}_topography.png`   |

Supports a `concurrent=True` flag — when set, the terrain plot step waits for Sentinel-2 data to become available (for use alongside the main pipeline).

### Concurrent Orchestrator (`get_outputs.py`)

Runs the Sentinel-2 pipeline and the environmental pipeline **in parallel threads**, displaying both Rich progress tables side-by-side. This is the fastest way to produce all outputs for a query.

---

## Setup

### Prerequisites

- Conda (Miniconda). Then install the env.yaml.
- `ffmpeg` on PATH (for video encoding)
- TERN API key (optional — only needed for SLGA soil downloads)

### Installation

```bash
# Create and activate the conda environment
conda env create -f paddock-ts-env.yml
conda activate paddock-ts

# Install fractionalcover3 from source (required for vegfrac)
# See: https://github.com/ANU-WALD/fractionalcover3

# Install PaddockTS in editable mode
pip install -e .
```

### Key Dependencies

| Category     | Packages                                                     |
|-------------|--------------------------------------------------------------|
| Geospatial  | rasterio, rioxarray, geopandas, shapely, odc-stac, pysheds  |
| Data        | xarray, numpy, scipy, pandas, dask                          |
| ML          | PyTorch, samgeo, scikit-learn, fractionalcover3, phenolopy   |
| Visualisation | matplotlib, opencv, Pillow, Rich                           |
| Remote sensing | pystac-client, odc-stac                                  |

---

## Usage

### Python API

```python
from datetime import date
from PaddockTS.query import Query
from PaddockTS.sentinel2_to_paddockTS_pipeline import run

# Define a query: bounding box [lon_min, lat_min, lon_max, lat_max] + date range
q = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2020, 1, 1),
    end=date(2024, 12, 31),
)

# Run the full pipeline
run(q)

# Force re-download and reprocessing
run(q, reload=True)

# Run environmental data pipeline only
from PaddockTS.run_environmental import run_environmental
run_environmental(q)

# Run everything concurrently (S2 pipeline + environmental in parallel)
from PaddockTS.get_outputs import get_outputs
get_outputs(q)
get_outputs(q, reload=True)  # clear cache and rerun
```

### Command Line

```bash
# Run with the built-in example query (NSW, 2020–2024)
python -m PaddockTS.sentinel2_to_paddockTS_pipeline

# Reload from scratch
python -m PaddockTS.sentinel2_to_paddockTS_pipeline --reload

# Environmental pipeline only
python -m PaddockTS.run_environmental

# Both pipelines concurrently (fastest)
python -m PaddockTS.get_outputs
```

### Checking Output Status

```python
from PaddockTS.status import status
status(q)
# {'sentinel2_video': True, 'sentinel2_paddocks_video': True, ...}
```

---

## Configuration

PaddockTS reads optional configuration from `~/.config/PaddockTS.json`:

```json
{
    "out_dir": "/path/to/outputs",
    "tmp_dir": "/path/to/intermediates",
    "email": "user@example.com",
    "tern_api_key": "your-tern-api-key"
}
```

| Field          | Default                           | Purpose                                |
|---------------|-----------------------------------|----------------------------------------|
| `out_dir`     | `~/Documents/PaddockTS-Outputs`   | Final outputs (videos, plots, PDFs)    |
| `tmp_dir`     | `~/Downloads/PaddockTS-Tmp`       | Intermediate caches (zarr, gpkg, tif)  |
| `email`       | `None`                            | Required for SILO API access           |
| `tern_api_key`| `None`                            | Required for SLGA soil data downloads  |

Outputs are organised by **stub** — a deterministic SHA-256 hash of `bbox + start + end`, ensuring identical queries always map to the same directory.

---

## Module Reference

### Query & Config

**`PaddockTS/query.py`** — Immutable `Query` dataclass built with `attrs`:
- Inputs: `bbox` (4 floats in WGS84), `start`/`end` (date objects)
- Derived: `stub` (SHA-256 hash), `tmp_dir`, `out_dir`, `sentinel2_path`, `vegfrac_path`, `centre_lon`, `centre_lat`

**`PaddockTS/config.py`** — Frozen `Config` object loaded once at import time. Reads `~/.config/PaddockTS.json` if it exists, otherwise uses defaults.

---

### Sentinel-2 Download

**`PaddockTS/Sentinel2/download_sentinel2.py`**

Queries the [DEA (Digital Earth Australia) STAC API](https://explorer.dea.ga.gov.au/stac) for Geoscience Australia's Sentinel-2 Analysis Ready Data (ARD), collection `ga_s2am_ard_3` / `ga_s2bm_ard_3`.

- Loads 11 spectral bands at 10 m resolution in EPSG:6933 (Albers Equal Area)
- Cloud-masks using the `oa_fmask` layer (removes cloud, cloud shadow, snow)
- Filters out frames with >20% NaN pixels
- Persists to Zarr with chunking along time, x, y axes
- Uses Dask distributed for parallel download

---

### Indices & Fractional Cover

**`PaddockTS/IndicesAndVegFrac/indices.py`** — Computes five vegetation/soil indices:

| Index | Formula | Purpose |
|-------|---------|---------|
| NDVI  | (NIR − Red) / (NIR + Red) | Green vegetation vigour |
| CFI   | NDVI × (R + 2G − B) | Canopy fluorescence proxy |
| NIRv  | NDVI × NIR | Near-infrared radiance of vegetation |
| NDTI  | (SWIR2 − SWIR3) / (SWIR2 + SWIR3) | Tillage / crop residue |
| CAI   | 0.5×(SWIR2 + SWIR3) − NIR | Cellulose absorption |

**`PaddockTS/IndicesAndVegFrac/veg_frac.py`** — Spectral unmixing via `fractionalcover3`:
- Decomposes each pixel into three fractions: bare ground (bg), photosynthetic vegetation (pv), non-photosynthetic vegetation (npv)
- Processes per-timestep, outputs to `{stub}_vegfrac.zarr`

---

### Paddock Segmentation

Three segmentation backends are available:

**v1 — SAMGeo** (`PaddockSegmentation/`) — *Active default in pipeline*
- Pre-segments using NDWI Fourier decomposition into a 3-band uint8 GeoTIFF
- Runs Meta's Segment Anything Model (ViT-H) via `samgeo`
- Vectorises mask raster, filters polygons by area (1–500 ha) and compactness

**v2 — K-Means** (`PaddockSegmentation2/`)
- Time-series K-Means clustering on NDVI + NDWI features
- Automatic cluster count selection via silhouette / variance ratio scoring
- Extracts boundaries via OpenCV contours or rasterio vectorisation

**v3 — W-Net** (`PaddockSegmentation3/`) — *Experimental*
- Unsupervised deep learning segmentation using a dual-UNet (W-Net)
- Trained end-to-end with Normalised Cut loss (no labels required)
- Encoder produces soft segmentation, decoder reconstructs input for regularisation

All backends output a GeoPackage (`.gpkg`) with polygon geometries, paddock IDs, area, and compactness.

---

### PaddockTS Time Series

**`PaddockTS/PaddockTS/make_paddockTS.py`**
- Rasterises paddock polygons into an integer mask aligned with the Sentinel-2 grid
- Computes the **median** of each spectral band/index per paddock per timestep
- Outputs an xarray Dataset with dimensions `(paddock, time)`, saved to `{stub}_paddockTS.zarr`

**`PaddockTS/PaddockTS/make_smoothed_paddockTS.py`**
- Resamples to regular N-day intervals (default 10 days)
- Fills gaps via PCHIP (monotone cubic) interpolation
- Smooths with Savitzky-Golay filter (window=7, order=2)

**`PaddockTS/PaddockTS/make_yearly_paddockTS.py`**
- Splits paddockTS into per-year datasets with a `doy` (day-of-year) coordinate
- Enables year-over-year phenological comparison

---

### Phenology

**`PaddockTS/Phenology/estimate_phenology.py`**
- Uses the [phenolopy](https://github.com/lewistrotter/Phenolopy) library to extract phenological metrics from yearly NDVI time series
- Computes per paddock per year:
  - **SoS** (Start of Season): onset of green-up
  - **PoS** (Peak of Season): maximum greenness
  - **EoS** (End of Season): senescence completion
  - Plus amplitude, growth rate, number of seasons

---

### Plotting & Reporting

| Module | Description | Output |
|--------|-------------|--------|
| `sentinel2_video.py` | RGB composite video with date overlay | `.mp4` |
| `vegfrac_video.py` | Fractional cover RGB (R=bare, G=green, B=dry) | `.mp4` |
| `sentinel2_paddocks_video.py` | Sentinel-2 + red paddock boundary overlay | `.mp4` |
| `vegfrac_paddocks_video.py` | VegFrac + paddock boundary overlay | `.mp4` |
| `calendar_plot.py` | Per-year grid: rows=paddocks, cols=~4 obs/month | `.png` per year |
| `phenology_plot.py` | NDVI scatter + smoothed curve + SoS/PoS/EoS markers | `.png` |
| `silo_plot.py` | SILO climate variables (temp, rain, radiation, etc.) | `.png` per group |
| `ozwald_plot.py` | OzWALD daily + 8-day climate/veg products | `.png` per group |
| `terrain_tiles_plot.py` | Elevation, slope, aspect, flow accumulation | `.png` |
| `make_pdf.py` | Assembles all plots into a single PDF report | `.pdf` |

All videos are encoded with H.264 via `ffmpeg` subprocess. Calendar plots use PIL for efficient large-image compositing.

---

### Environmental Data

Standalone modules for downloading Australian environmental datasets. These are **not** part of the main pipeline loop but can be invoked independently for a given `Query`.

| Source | Module | Data | Temporal | Spatial |
|--------|--------|------|----------|---------|
| [SILO](https://www.longpaddock.qld.gov.au/silo/) | `Environmental/SILO/` | 18 climate variables (rainfall, temperature, radiation, ET, humidity) | Daily | Point (nearest grid cell) |
| [OzWALD](https://www.csiro.au/ozwald) | `Environmental/OzWALD/` | 9 daily meteo + 13 eight-day products (LAI, GPP, NDVI, soil moisture) | Daily / 8-day | Point (nearest grid cell) |
| [Copernicus DEM](https://registry.opendata.aws/copernicus-dem/) | `Environmental/TerrainTiles/` | 30m elevation, slope, aspect, flow accumulation, TWI | Static | Raster (full bbox) |
| [SLGA](https://www.clw.csiro.au/aclep/soilandlandscapegrid/) | `Environmental/SLGASoils/` | 11 soil attributes at 6 depth ranges (clay, sand, pH, OC, bulk density, etc.) | Static | Raster (full bbox) |

---

## Data Flow

```
Query
  │
  ├── tmp_dir/{stub}/                          Intermediate cache
  │   ├── {stub}_sentinel2.zarr                11-band cloud-masked imagery
  │   ├── {stub}_vegfrac.zarr                  Fractional cover (bg, pv, npv)
  │   ├── {stub}_preseg.tif                    Presegmentation features
  │   ├── {stub}_sam_mask.tif                  SAM segmentation mask
  │   ├── {stub}_sam_raw.gpkg                  Raw SAM polygons (unfiltered)
  │   ├── {stub}_paddocks.gpkg                 Final paddock boundaries
  │   ├── {stub}_paddockTS.zarr                Per-paddock time series
  │   ├── {stub}_paddockTS_smoothed.zarr       Smoothed time series
  │   ├── {stub}_paddockTS_{year}.zarr         Per-year time series
  │   ├── {stub}_silo.csv                      SILO climate data
  │   ├── {stub}_ozwald_daily.csv              OzWALD daily data
  │   ├── {stub}_ozwald_8day.csv               OzWALD 8-day data
  │   ├── {stub}_terrain.tif                   DEM raster
  │   └── {stub}_sentinel2_to_paddockTS.log    Pipeline log
  │
  └── out_dir/{stub}/                          Final outputs
      ├── {stub}_sentinel2.mp4                 Sentinel-2 RGB video
      ├── {stub}_sentinel2_paddocks.mp4        Sentinel-2 + paddocks video
      ├── {stub}_vegfrac.mp4                   Fractional cover video
      ├── {stub}_vegfrac_paddocks.mp4          VegFrac + paddocks video
      ├── {stub}_calendar_{year}.png           Calendar thumbnail grids
      ├── {stub}_phenology.png                 Phenology plot
      ├── {stub}_topography.png                Terrain analysis plot
      ├── {stub}_silo_*.png                    SILO climate plots
      ├── {stub}_ozwald_*.png                  OzWALD plots
      └── {stub}_report.pdf                    Composite PDF report
```

---

## Output Artifacts

### Videos (`.mp4`)
- **Sentinel-2 RGB**: True-colour composite, contrast-enhanced (×3), date stamped
- **VegFrac RGB**: R = bare ground, G = photosynthetic vegetation, B = non-photosynthetic vegetation
- **+Paddocks variants**: Red boundary lines with paddock ID labels overlaid

### Calendar Plots (`.png`)
- One image per year
- Rows = paddocks (sorted largest-first), columns = ~48 observation slots (4 per month)
- Each cell is a cropped RGB thumbnail of the paddock at that timestep

### Phenology Plot (`.png`)
- Grid of subplots: rows = paddocks, columns = years
- Raw observations (scatter), smoothed NDVI curve, SoS/PoS/EoS vertical markers

### PDF Report
- Combines: landscape videos (as frames), SILO/OzWALD climate plots, calendar grids, phenology plots, terrain analysis

---

## Running Individual Modules

Each module includes a `test()` function and `__main__` block that uses the example query (NSW, 2020–2024):

```bash
# Sentinel-2 download
python -m PaddockTS.Sentinel2.download_sentinel2

# Spectral indices
python -m PaddockTS.IndicesAndVegFrac.indices

# Fractional cover
python -m PaddockTS.IndicesAndVegFrac.veg_frac

# Paddock segmentation (SAMGeo)
python -m PaddockTS.PaddockSegmentation.get_paddocks

# Paddock segmentation (K-Means)
python -m PaddockTS.PaddockSegmentation2.get_paddocks

# Paddock segmentation (W-Net)
python -m PaddockTS.PaddockSegmentation3.get_paddocks

# Videos
python -m PaddockTS.Plotting.sentinel2_video
python -m PaddockTS.Plotting.vegfrac_video

# Environmental data
python -m PaddockTS.Environmental.SILO.download_silo
python -m PaddockTS.Environmental.OzWALD.download_ozwald_daily
python -m PaddockTS.Environmental.TerrainTiles.download_terrain_tiles
python -m PaddockTS.Environmental.SLGASoils.download_slgasoils
```

---

## License

MIT
