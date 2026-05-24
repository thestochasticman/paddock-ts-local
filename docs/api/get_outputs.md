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

To delete all caches under `query.tmp_dir` and `query.out_dir` and
force a full rebuild:

```python
get_outputs(q, reload=True)
```

This is rarely needed — the per-stage `_SUCCESS` markers catch partial
writes automatically. Use it when you've changed something the cache
doesn't track (e.g. tweaking a SAM filter threshold) and you want a
clean run.

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

The terrain plot in the environmental thread synchronously waits for
`sentinel2_clean.zarr` to appear (it overlays the terrain rendering on
the S2 grid extent). If the Sentinel-2 thread fails before producing
the clean cube, the terrain plot raises a clear `RuntimeError` instead
of hanging forever.

---

## Reference

::: PaddockTS.get_outputs.get_outputs
