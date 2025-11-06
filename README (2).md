# PaddockTS Local

A Python toolkit for paddock-level remote sensing workflows. It helps you:

- Query an area of interest and time range  
- Download satellite & environmental data  
- Compute vegetation indices and fractional cover  
- Segment paddock boundaries  
- Build paddock time-series and plot maps/checkpoints

> This repo is a light, local developer workspace for the core library under [`PaddockTS/`](./PaddockTS/) plus configs, demos, and utilities.

---

## Installation

Using Conda (recommended):

```bash
conda env create -f env.yml
conda activate PaddockTSEnv
```

Minimal editable install with pip:

```bash
pip install -e .
```

> Dependencies/env specs live in [`env.yml`](./env.yml), [`env_2.yml`](./env_2.yml), and [`pyproject.toml`](./pyproject.toml).

---

## Quick start

```python
from PaddockTS.query import Query
from PaddockTS.get_outputs import get_outputs


# 1) Define your area/time window
q = Query(
    lat=-33.5040,
    lon=148.4,
    buffer=0.01,
    start_time=date(2020, 1, 1),
    end_time=date(2020, 6, 1),
    collections=['ga_s2am_ard_3', 'ga_s2bm_ard_3'],
    bands=[
        'nbart_blue',
        'nbart_green',
        'nbart_red',
    ],
    filter=Filter.lt('eo:cloud_cover', 10)
)

# 2) Produce standard outputs
outputs = get_outputs(q)
```

This typically will:

- Download Sentinel-2 and environmental data  
- Compute indices and fractional cover  
- Segment paddock boundaries  
- Generate paddock-level time-series and optional figures

---

## Configuration

On first import, [`PaddockTS/legend.py`](./PaddockTS/legend.py) creates `~/.configs/PaddockTSLocal.json` with:

- `out_dir`: final outputs  
- `tmp_dir`: intermediates (stacks, shapefiles)  
- `scratch_dir`: heavy/temporary files

Update these paths after first run to match your storage.

---

## Repository layout

### Top-level files & folders

- [`PaddockTS/`](./PaddockTS/) — Core Python package (details below).  
- [`demo/`](./demo/) — Simple examples/walk-throughs. 
- [`env.yml`](./env.yml) — Primary Conda environment.  
- [`env_2.yml`](./env_2.yml) — Alternative/experimental environment.  
- [`pyproject.toml`](./pyproject.toml) — Project metadata & dependencies (PEP 621). 
- [`test.py`](./test.py) — Small local test harness.  
- [`README.md`](./README.md) — This document.

---

### Python package: `PaddockTS/`

#### Data acquisition

- [`PaddockTS/Data/download_ds2.py`](./PaddockTS/Data/download_ds2.py)  
  Sentinel-2 acquisition: STAC-style search, AOI/time filtering, fetches required bands.

- [`PaddockTS/Data/environmental.py`](./PaddockTS/Data/environmental.py)  
  Environmental data acquisition (e.g., rainfall/temperature). Stages variables for later joins.

#### Indices & fractional cover

- [`PaddockTS/IndicesAndVegFrac/indices.py`](./PaddockTS/IndicesAndVegFrac/indices.py)  
  Spectral indices (e.g., NDVI, NDWI). Extend as needed.

- [`PaddockTS/IndicesAndVegFrac/veg_frac.py`](./PaddockTS/IndicesAndVegFrac/veg_frac.py)  
  Fractional cover from pretrained model/features.

- [`PaddockTS/IndicesAndVegFrac/add_indices_and_veg_frac.py`](./PaddockTS/IndicesAndVegFrac/add_indices_and_veg_frac.py)  
  Orchestrates index + veg-frac computation for scenes/stacks.

- [`PaddockTS/IndicesAndVegFrac/utils.py`](./PaddockTS/IndicesAndVegFrac/utils.py)  
  Band math, masking, QA, and shared I/O helpers.

#### Paddock segmentation

- [`PaddockTS/PaddockSegmentation/_1_presegment.py`](./PaddockTS/PaddockSegmentation/_1_presegment.py)  
  Builds NDWI time-series and Fourier (seasonality) representation for segmentation.

- [`PaddockTS/PaddockSegmentation/_2_segment.py`](./PaddockTS/PaddockSegmentation/_2_segment.py)  
  Consumes Fourier image, produces paddock masks/polygons.

- [`PaddockTS/PaddockSegmentation/segment_paddocks.py`](./PaddockTS/PaddockSegmentation/segment_paddocks.py)  
  High-level driver that runs pre-segment + segment for an AOI.

- [`PaddockTS/PaddockSegmentation/utils.py`](./PaddockTS/PaddockSegmentation/utils.py)  
  Morphology, polygonization, cleaning, and other helpers.

#### Paddock time-series

- [`PaddockTS/PaddockTS/get_paddock_ts.py`](./PaddockTS/PaddockTS/get_paddock_ts.py)  
  Aggregates per-pixel indices/cover within paddock polygons over time; can join environmental variables.

#### Plotting

- [`PaddockTS/Plotting/plotting_functions.py`](./PaddockTS/Plotting/plotting_functions.py)  
  General plotting utilities for maps/time-series.

- [`PaddockTS/Plotting/checkpoint_plots.py`](./PaddockTS/Plotting/checkpoint_plots.py)  
  Visualizes intermediate “checkpoints” (masks, overlays, diagnostic panels).

- [`PaddockTS/Plotting/topographic_plots.py`](./PaddockTS/Plotting/topographic_plots.py)  
  Hillshade, slope/aspect overlays and topography QA plots.

#### Core helpers

- [`PaddockTS/filter.py`](./PaddockTS/filter.py)  
  STAC/collection filter builder used by download routines.

- [`PaddockTS/legend.py`](./PaddockTS/legend.py)  
  Configuration manager: initializes and reads `~/.configs/PaddockTSLocal.json`.

- [`PaddockTS/query.py`](./PaddockTS/query.py)  
  Defines the `Query` object (AOI, CRS, date range, naming).

- [`PaddockTS/get_outputs.py`](./PaddockTS/get_outputs.py)  
  Convenience entry point tying the high-level steps together for a `Query`.

- [`PaddockTS/__init__.py`](./PaddockTS/__init__.py)  
  Package marker; may expose selected top-level imports.

---

## Development notes

- Start with a small AOI and short date window to validate your environment and caches.  
- Point `gdalwmscache/`, `tmp_dir`, and `scratch_dir` to fast local storage—these can grow quickly.  
- If you add indices or environmental variables, wire changes through:
  1) acquisition ([`PaddockTS/Data/...`](./PaddockTS/Data/)) →  
  2) transforms ([`PaddockTS/IndicesAndVegFrac/...`](./PaddockTS/IndicesAndVegFrac/)) →  
  3) aggregation ([`PaddockTS/PaddockTS/get_paddock_ts.py`](./PaddockTS/PaddockTS/get_paddock_ts.py)) →  
  4) plotting ([`PaddockTS/Plotting/...`](./PaddockTS/Plotting/)).

---

## License

TBD (add your license here).

---

**Attribution**  
The file list and module purposes reflect the current repository structure. Update this README if files are renamed or moved.
