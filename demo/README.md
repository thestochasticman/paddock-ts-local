# PaddockTS demos

Runnable Jupyter notebooks showing how to use the system end-to-end
and how to mine its outputs. Start with
[`01_quickstart.ipynb`](01_quickstart.ipynb) and then pick the others
based on what you need.

### Workflow notebooks — *how to run the pipeline*

| Notebook | What it covers |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | The simplest end-to-end run: bbox + dates → `get_outputs(query)` → live dashboard → review the calendar / phenology / PDF. |
| [`02_pipeline_stages.ipynb`](02_pipeline_stages.ipynb) | Calling each stage individually (Sentinel-2 download, clean, indices, fractional cover, SAM segmentation, per-paddock TS, phenology) — useful for debugging, fine-tuning, or replacing a single stage with your own implementation. |
| [`03_custom_paddocks.ipynb`](03_custom_paddocks.ipynb) | Bringing your own paddock boundaries (GeoPackage / Shapefile / GeoJSON) and skipping SAM. Uses the bundled `artifacts/PaddockSet1.gpkg` as the worked example. |

### Inspection notebooks — *what to do with the outputs*

Ordered to follow the pipeline output order: raw S2 / fractional-cover
videos → paddock polygons → per-paddock time series → phenology
metrics → the stitched PDF report. Each notebook assumes you've
finished a pipeline run for the query in its config cell (or you've
edited that cell to point at your own run).

| Notebook | What it covers |
|---|---|
| [`04_inspect_videos.ipynb`](04_inspect_videos.ipynb) | Embed the Sentinel-2 / fractional-cover / paddocks-overlay MP4s inline and view the per-paddock calendar PNGs. |
| [`05_inspect_paddocks.ipynb`](05_inspect_paddocks.ipynb) | Load the paddocks GeoPackage, summarise by area / compactness, filter, join with per-paddock NDVI from the time-series cube, and produce thematic maps. |
| [`06_inspect_time_series.ipynb`](06_inspect_time_series.ipynb) | The `(paddock, time)` Zarr cube. Slice by paddock / index / year, compare paddocks, compare indices, AOI-average plots, raw vs Savitzky-Golay-smoothed traces. |
| [`07_inspect_phenology.ipynb`](07_inspect_phenology.ipynb) | Per-paddock SoS / PoS / EoS / amplitude / integral metrics. Distributions, outlier detection, year-over-year comparison, and overlaying SoS / PoS / EoS markers on a paddock's NDVI trace. |
| [`08_inspect_pdf.ipynb`](08_inspect_pdf.ipynb) | Locate the stitched A4 PDF report, read its embedded metadata, and preview it inline — either as an in-browser `<iframe>` or as page-by-page PNGs via `pdf2image`. |

## Running the notebooks

```bash
# From the repo root, with the paddockts conda env activated:
jupyter lab demo/
```

Each notebook is keyed to its own `stub` so the demos don't collide
with each other on disk. They use small AOIs (~2 km × 2 km) and short
date windows (a few months) so cold runs finish in a few minutes; the
content-addressed cache means rerunning is instant.

## Prerequisites

- PaddockTS installed (`pip install -e .` or `conda env create -f
  paddock-ts-env.yml && conda activate paddockts && pip install -e .`).
- Internet access — Sentinel-2 is downloaded on demand from Geoscience
  Australia's STAC.
- **No** credentials needed for the Sentinel-2 chain. SILO needs an
  email (any working address registered with the service) and SLGA
  needs a TERN API key — see the [Getting Started docs](https://thestochasticman.github.io/paddock-ts-local/getting-started/)
  if you plan to use those environmental layers.

## What gets written to disk

Each notebook prints the relevant paths as it goes. By default:

- **Outputs** (paddocks gpkg, time-series zarr, calendar PNGs, MP4
  videos, PDF report) → `~/Documents/PaddockTS-Outputs/<stub>/`
- **Caches** (raw + clean Sentinel-2 zarr, indices, fractional cover,
  presegmentation tif, SAM masks, terrain DEM, env CSVs) →
  `~/Downloads/PaddockTS-Tmp/...`

To redirect both, set `out_dir` / `tmp_dir` in `~/.config/PaddockTS.json`
before running. See [Getting Started → Configure](https://thestochasticman.github.io/paddock-ts-local/getting-started/#configure).
