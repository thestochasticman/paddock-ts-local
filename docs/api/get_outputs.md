# Pipeline driver

`get_outputs` is the orchestrator for the full PaddockTS run. It
spawns two worker threads — one for environmental data (terrain,
OzWALD, SILO, SLGA) and one for the Sentinel-2 → PaddockTS chain —
and renders a live two-column status dashboard while they run.

All stdout, stderr, Python `logging`, `warnings.warn`, and even
C-level fd 2 writes from GDAL / PROJ / TFLite are captured into a
bounded ring buffer and (optionally) shown in a log panel below the
tables. The captured output is restored after the dashboard tears
down, so anything written during the run is preserved in the parent
terminal scrollback.

---

## Basic usage

```python
from datetime import date
from PaddockTS.query import Query
from PaddockTS.get_outputs import get_outputs

q = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2024, 1, 1),
    end=date(2024, 12, 31),
    stub="run_demo",
)

get_outputs(q)
```

You'll see something like:

```text
┌── Environmental ─────────────┐  ┌── Sentinel-2 → PaddockTS ────────────┐
│ #  Step               Status │  │ #  Step                       Status │
│ 1  Download terrain  ✓ done  │  │ 1  Download Sentinel-2       ✓ done  │
│ 2  Download OzWALD   ✓ done  │  │ 2  Compute indices           ✓ done  │
│ 3  Download SILO     skipped │  │ 3  Compute fractional cover  running │
│ 4  Download SLGA     skipped │  │ 4  Sentinel-2 video          pending │
│ 5  OzWALD plot       running │  │ 5  Segment paddocks (SAM)    pending │
│ 6  SILO plot         skipped │  │ ...                                  │
│ 7  Terrain plot      waiting │  │                                      │
└──────────────────────────────┘  └──────────────────────────────────────┘
```

---

## Reload from scratch

To wipe every cached artifact for this query and force a clean rebuild:

```python
get_outputs(q, reload=True)
```

This deletes:

- **`query.query_dir`** — per-(bbox, time) caches: Sentinel-2 raw +
  clean zarrs, indices zarr, fractional cover zarr, presegmentation
  tif, SAM mask + raw polygons + filtered paddocks gpkg.
- **`query.terrain_path`** — the Copernicus DEM tile (per-bbox,
  time-invariant). The wider `aoi_dir` is left alone so other queries
  with the same bbox but a different time range aren't surprised.
- **`query.tmp_dir`** — per-stub time-series zarrs (paddockTS,
  yearly, smoothed).
- **`query.out_dir`** — every final output (PNGs, MP4s, PDF report).

This is rarely needed — the per-stage `_SUCCESS` markers catch partial
writes automatically. Use it when you've changed something the cache
doesn't track (e.g. tweaking a SAM filter threshold) and want a clean
run.

---

## Show the log panel

By default the dashboard only shows the status tables. Pass
`show_log=True` to render a tail-of-log panel below the tables —
useful for debugging stuck steps:

```python
get_outputs(q, show_log=True)
```

---

## Bring your own paddocks

Skip SAM segmentation entirely and use a user-provided paddocks file
(GeoPackage, Shapefile, or GeoJSON) instead. The downstream stages
(per-paddock TS, videos with overlays, calendar, phenology, PDF) all
run on your paddocks.

```python
get_outputs(
    q,
    paddocks_filepath="/path/to/paddocks.gpkg",
    skip_sam=True,
    label_col="paddock_name",   # column for human-readable labels
)
```

Or, run **both** SAM and user paddocks in the same call — useful for
comparison. Just drop `skip_sam=True`:

```python
get_outputs(
    q,
    paddocks_filepath="/path/to/paddocks.gpkg",
    label_col="paddock_name",
)
```

The driver will run every paddock-dependent stage twice — once with
SAM paddocks, once with yours — and the PDF report will include both
sets of calendars and phenology plots.

---

## End-to-end with a paddocks-defined AOI

If your paddocks file already defines the area of interest, use
`Query.build_from_paddocks` to take its envelope as the bbox:

```python
from PaddockTS.query import Query

paddocks_fp = "/path/to/paddocks.gpkg"

q = Query.build_from_paddocks(
    paddocks_filepath=paddocks_fp,
    start=date(2024, 1, 1),
    end=date(2024, 12, 31),
    stub="my_farm",
    label_col="paddock_name",
)

get_outputs(
    q,
    paddocks_filepath=paddocks_fp,
    skip_sam=True,
    label_col="paddock_name",
)
```

---

## Error handling

If a stage fails, its row turns red in the dashboard with the time
elapsed before the failure. The other thread keeps running so partial
progress is preserved. After both threads finish, `get_outputs`
re-raises the first exception encountered.

### Cascading-skip for dependent steps

When an upstream step errors (or is itself skipped), every downstream
step that depends on it is marked `skipped` rather than running,
failing on a missing input, and lighting up the dashboard with a wall
of red. The dependency map (`_S2_STEP_DEPS`) covers the SAM chain
(steps 5, 8, 10, 12, 14, 16, 18 all depend on step 4 producing a
valid paddocks gpkg; the TS / yearly / phenology / plot stages
further depend on each other in order) and the user-paddocks TS chain
(steps 13 / 15 / 19 depend on step 11 having produced a valid user
paddockTS zarr).

In practice this means: if SAM segmentation crashes, you'll see one
red row at step 4 and skipped (cyan) rows for the seven SAM-dependent
steps that follow — making the actual failure point obvious instead
of buried.

### Missing-credentials skip

Environmental steps that require credentials are silently skipped
when the credential isn't configured, rather than raising:

- Step 3 (Download SILO) and step 6 (SILO plot) → skipped if
  `config.email` is unset.
- Step 4 (Download SLGA soils) → skipped if `config.tern_api_key` is
  unset.

Other env steps (terrain, OzWALD) and the entire Sentinel-2 chain
work without any credentials.

### Terrain-plot wait guard

The terrain plot in the environmental thread synchronously waits for
`sentinel2_clean.zarr` to appear (it overlays the terrain rendering on
the S2 grid extent). If the Sentinel-2 thread fails before producing
the clean cube, the terrain plot raises a clear `RuntimeError` instead
of hanging forever.

---

## Reference

::: PaddockTS.get_outputs.get_outputs
