# PaddockTS

Satellite imagery to paddock boundaries and time-series videos.

Downloads Sentinel-2 data for a given bounding box and date range, computes vegetation indices and fractional cover, segments paddock boundaries, and generates video outputs.

## Setup

```bash
conda env create -f paddock-ts-env.yml
conda activate paddock-ts
pip install -e .
```

`fractionalcover3` needs to be installed from source separately.

## Usage

```python
from datetime import date
from PaddockTS.query import Query
from PaddockTS.sentinel2_to_paddock_pipeline import run

q = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2020, 1, 1),
    end=date(2024, 12, 31),
)

run(q)
run(q, reload=True)  # delete cached data and rerun
```

Or from the command line:

```bash
python -m PaddockTS.sentinel2_to_paddock_pipeline
python -m PaddockTS.sentinel2_to_paddock_pipeline --reload
```

## Pipeline steps

1. Download Sentinel-2 imagery (DEA STAC) -> `{stub}_sentinel2.zarr`
2. Compute indices (NDVI, CFI, NIRv, NDTI, CAI) -> in-memory
3. Compute fractional cover (bg, pv, npv) -> `{stub}_vegfrac.zarr`
4. Segment paddocks (k-means + vectorization) -> `{stub}_paddocks.gpkg`
5. Sentinel-2 video -> `{stub}_sentinel2.mp4`
6. Sentinel-2 + paddocks video -> `{stub}_sentinel2_paddocks.mp4`
7. Vegfrac video -> `{stub}_vegfrac.mp4`
8. Vegfrac + paddocks video -> `{stub}_vegfrac_paddocks.mp4`

Intermediates go to `~/Downloads/PaddockTS-Tmp/{stub}/`, outputs to `~/Documents/PaddockTS-Outputs/{stub}/`. Override via `~/.config/PaddockTS.json`.

`{stub}` is a SHA-256 hash of `bbox + start + end`.

## Check status

```python
from PaddockTS.status import status
status(q)
# {'sentinel2_video': True, 'sentinel2_paddocks_video': True, ...}
```

## Project structure

- `PaddockTS/query.py` — Query dataclass (bbox, start, end -> stub, paths)
- `PaddockTS/config.py` — out_dir, tmp_dir, silo_dir
- `PaddockTS/sentinel2_to_paddock_pipeline.py` — runs everything with a Rich progress table
- `PaddockTS/status.py` — check which outputs exist for a query
- `PaddockTS/Sentinel2/` — Sentinel-2 download (STAC search, cloud masking, Zarr)
- `PaddockTS/IndicesAndVegFrac/` — spectral indices and fractional cover
- `PaddockTS/PaddockSegmentation2/` — k-means based paddock segmentation
- `PaddockTS/PaddockSegmentation3/` — experimental W-Net (PyTorch) segmentation
- `PaddockTS/Plotting/` — video generation (sentinel2, vegfrac, with/without paddock overlays)
- `PaddockTS/Environmental/` — SLGA soils and terrain tile downloads (standalone)

## Running individual modules

Each module has a `test()` function and `__main__` block that uses the example query (NSW, 2020-2024):

```bash
python -m PaddockTS.Sentinel2.download_sentinel2
python -m PaddockTS.IndicesAndVegFrac.indices
python -m PaddockTS.IndicesAndVegFrac.veg_frac
python -m PaddockTS.PaddockSegmentation2.get_paddocks
python -m PaddockTS.Plotting.sentinel2_video
python -m PaddockTS.Plotting.vegfrac_video
```
