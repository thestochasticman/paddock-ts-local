# PaddockTS

**Paddock segmentation and time-series analysis from Sentinel-2 imagery.**

PaddockTS turns a bounding box and a date range into a full set of
paddock-scale outputs: segmented paddock polygons, vegetation indices,
fractional cover, per-paddock time series, phenology metrics, and
ready-to-use plots and videos.

📚 **Documentation:** <https://thestochasticman.github.io/paddock-ts-local/>

---

## Install

### Conda (recommended)

```bash
git clone https://github.com/thestochasticman/paddock-ts-local.git
cd paddock-ts-local
conda env create -f paddock-ts-env.yml
conda activate paddockts-env
pip install -e .
```

### pip

```bash
pip install PaddockTS
```

Requires Python ≥ 3.11 and a working geospatial native stack (GDAL,
PROJ, GEOS). Conda is the easiest path to that.

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

This runs the full pipeline (Sentinel-2 download → indices → fractional
cover → SAM paddock segmentation → time series → phenology → plots) plus
the environmental data pipeline (terrain, climate, soils) in parallel,
with a live status dashboard.

Outputs land under `~/Documents/PaddockTS-Outputs/<stub>/` by default —
configurable via `~/.config/PaddockTS.json`.

---

## Pipeline at a glance

| Sentinel-2 → PaddockTS (13 stages) | Environmental (7 stages) |
|---|---|
| Download Sentinel-2 | Download terrain |
| Compute indices | Download OzWALD daily |
| Compute fractional cover | Download SILO |
| Sentinel-2 video | Download SLGA soils |
| Segment paddocks (SAM) | OzWALD plot |
| Sentinel-2 + paddocks video | SILO plot |
| Fractional cover video | Terrain plot |
| Fractional cover + paddocks video | |
| Make paddockTS | |
| Make yearly paddockTS | |
| Estimate phenology | |
| Calendar plot | |
| Phenology plot | |

Every stage is independently callable; you don't have to use
`get_outputs` if you just want one piece. See the
[pipeline page](https://thestochasticman.github.io/paddock-ts-local/pipeline/)
for the full breakdown.

---

## License

MIT — see [LICENSE](LICENSE).

PaddockTS vendors third-party code under permissive licenses. See
[`PaddockTS/LICENSES/`](PaddockTS/LICENSES/) for attribution to:

- [`fractionalcover3`](https://github.com/jrsrp/fractionalcover3) by Robert Denham (MIT)
- [`phenolopy`](https://github.com/lewistrotter/phenolopy) by Lewis Trotter (Apache 2.0)
