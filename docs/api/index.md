# API reference

Public API for the PaddockTS package. Each page below covers one
stage of the pipeline plus its supporting types — prose, example
usage, and auto-generated signatures from the source docstrings.

| Module | What it does |
|---|---|
| [Query & Config](query.md) | The bbox + date-range object every stage consumes, plus runtime configuration. |
| [Sentinel-2 download](sentinel2.md) | STAC fetch, cloud mask, scene drop, and Zarr write. |
| [Spectral indices](spectral_indices.md) | NDVI, CFI, NIRv, NDTI, CAI from S2 reflectance. |
| [Fractional cover](fractional_cover.md) | bg / pv / npv unmixing via TFLite MLP. |
| [Paddock segmentation](paddock_segmentation.md) | SAM-based field boundary extraction. |
| [Paddock time series](paddock_time_series.md) | Per-paddock medians + yearly split + smoothing. |
| [Phenology](phenology.md) | Per-paddock seasonal metrics (SoS / PoS / EoS / amplitudes). |
| [Plotting](plotting.md) | Static plots + animation videos + PDF report. |
| [Environmental data](environmental.md) | Terrain, climate, soil downloads + their plots. |
| [Pipeline driver](get_outputs.md) | `get_outputs()` orchestrator and dashboard. |

## How these pages are generated

Each page combines:

- hand-written prose explaining the *what* and *why* of each module,
- runnable examples showing a typical call,
- auto-generated reference rendered by
  [mkdocstrings](https://mkdocstrings.github.io/) from the Python
  docstrings, type hints, and signatures in `PaddockTS/`.

To improve any auto-generated section, edit the docstring of the
underlying function or class — the page rebuilds on the next docs
build.

## Stability

The public API surface is anything importable from a top-level
PaddockTS subpackage (e.g.
`PaddockTS.FractionalCover.compute_fractional_cover`). Modules and
members prefixed with `_` (e.g. `PaddockTS.FractionalCover._unmix`,
`PaddockTS.Phenology._phenolopy`) are internal and may change
without notice.

## Shared contract: caching

Every stage on disk is guarded by a `_SUCCESS` marker written
**after** the data file completes. On a rerun, the stage:

1. Checks for the data file and the marker.
2. If both exist, loads from cache and returns immediately.
3. If the data exists but the marker is missing (kill mid-write,
   OOM, network drop), wipes the partial cache and re-runs cleanly.

This contract is identical across `download_sentinel2`,
`clean_sentinel2`, `compute_indices`, `compute_fractional_cover`,
`get_paddocks`, `make_paddock_time_series`,
`make_yearly_paddock_time_series`, `make_smoothed_paddock_time_series`,
and the terrain download.
