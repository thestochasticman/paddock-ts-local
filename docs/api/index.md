# API reference

Auto-generated reference for the public PaddockTS API. Each page below
covers one stage of the pipeline plus its supporting types.

| Module | What it does |
|---|---|
| [Query](query.md) | The bbox + date-range object every stage consumes. |
| [Sentinel-2 download](sentinel2.md) | STAC fetch + cloud mask + Zarr write. |
| [Spectral indices](spectral_indices.md) | NDVI, CFI, NIRv, NDTI, CAI from S2 reflectance. |
| [Fractional cover](fractional_cover.md) | bg / pv / npv unmixing via TFLite MLP. |
| [Paddock segmentation](paddock_segmentation.md) | SAM-based field boundary extraction. |
| [Paddock time series](paddock_ts.md) | Per-paddock medians per timestamp. |
| [Phenology](phenology.md) | Per-paddock seasonal metrics. |
| [Plotting](plotting.md) | Static plots + animation videos. |
| [Environmental data](environmental.md) | Terrain, climate, soil downloads. |
| [Pipeline driver](get_outputs.md) | `get_outputs()` orchestrator. |

## How the API is documented

Pages are generated from the source via
[mkdocstrings](https://mkdocstrings.github.io/) reading docstrings, type
hints, and signatures directly from `PaddockTS/`. To improve a page, edit
the docstring of the corresponding function or class.

## Stability

Public API surface is currently anything importable from a top-level
PaddockTS subpackage (e.g. `PaddockTS.FractionalCover.compute_fractional_cover`).
Modules and members prefixed with `_` (e.g. `PaddockTS.FractionalCover._unmix`,
`PaddockTS.Phenology._phenolopy`) are internal and may change without notice.
