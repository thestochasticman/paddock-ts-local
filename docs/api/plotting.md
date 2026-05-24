# Plotting

Static plots, animation videos, and a stitched PDF report — every
review-grade output PaddockTS produces.

| Function | Output | What it shows |
|---|---|---|
| `sentinel2_video` | `{stub}_sentinel2.mp4` | True-colour Sentinel-2 timeline, date stamped per frame. |
| `sentinel2_video_with_paddocks` | `{stem}_sentinel2_paddocks.mp4` | Same, with red paddock boundaries and IDs overlaid. |
| `fractional_cover_video` | `{stub}_fractional_cover.mp4` | False-colour cover timeline (R=bg, G=pv, B=npv). |
| `fractional_cover_paddocks_video` | `{stem}_fractional_cover_paddocks.mp4` | Same, with paddock overlays. |
| `calendar_plot` | `{stem}_calendar_<year>_p01.png` | Per-paddock thumbnail calendar (48 slots/year, paddock-masked). |
| `phenology_plot` | `{stem}_phenology_p01.png` | Per-paddock × per-year NDVI curves with SoS / PoS / EoS markers. |
| `ozwald_daily_plot` | `{stub}_ozwald_daily_*.png` | OzWALD climate panels (temperature, precipitation, wind, radiation). |
| `silo_plot` | `{stub}_silo_*.png` | SILO climate panels (temperature, rainfall, radiation, ET, humidity). |
| `terrain_tiles_plot` | `{stub}_topography.png` | 2 × 2 panel: elevation, flow accumulation, aspect, slope. |
| `make_pdf` | `{stub}.pdf` | Single PDF stitching every plot above, with section headers. |

All static plots and videos write to `{query.out_dir}/`.

---

## Sentinel-2 videos

Each frame is a normalised RGB composite from `nbart_red`,
`nbart_green`, `nbart_blue`, with the acquisition date stamped in the
top-right corner. Frames are written as PNGs to a temporary directory
then encoded to H.264 with `ffmpeg` (`libopenh264`). H.264 requires
even dimensions, so the final size is rounded down to even after
scaling to `min_size`.

### Example

```python
from datetime import date
from PaddockTS.query import Query
from PaddockTS.Plotting.sentinel2_video import sentinel2_video
from PaddockTS.Plotting.sentinel2_paddocks_video import sentinel2_video_with_paddocks

q = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2024, 1, 1),
    end=date(2024, 12, 31),
    stub="vid_demo",
)

sentinel2_video(q, fps=4, min_size=1080)
# -> {out_dir}/vid_demo_sentinel2.mp4

# With auto SAM paddocks overlaid
sentinel2_video_with_paddocks(q)
# -> {out_dir}/<sam_stem>_sentinel2_paddocks.mp4

# With user paddocks overlaid + custom labels
sentinel2_video_with_paddocks(
    q,
    paddocks_filepath="/path/to/paddocks.gpkg",
    label_col="paddock_name",
)
```

### Reference

::: PaddockTS.Plotting.sentinel2_video.sentinel2_video
::: PaddockTS.Plotting.sentinel2_paddocks_video.sentinel2_video_with_paddocks

---

## Fractional cover videos

Each frame is a false-colour composite mapping the three fractional
cover bands to RGB channels:

- **R = bg** (bare ground)
- **G = pv** (green vegetation)
- **B = npv** (non-green vegetation)

Fractions are renormalised to sum to 1 before display, so a fully bare
pixel is pure red, fully green vegetation is pure green, and so on.

### Example

```python
from PaddockTS.Plotting.fractional_cover_video import fractional_cover_video
from PaddockTS.Plotting.fractional_cover_paddocks_video import fractional_cover_paddocks_video

fractional_cover_video(q, fps=4, min_size=1080)
# -> {out_dir}/vid_demo_fractional_cover.mp4

fractional_cover_paddocks_video(q)
# -> {out_dir}/<sam_stem>_fractional_cover_paddocks.mp4
```

### Reference

::: PaddockTS.Plotting.fractional_cover_video.fractional_cover_video
::: PaddockTS.Plotting.fractional_cover_paddocks_video.fractional_cover_paddocks_video

---

## Diagnostic plots

### Calendar plot

One PNG per year (split into pages if there are many paddocks). Rows
are paddocks (largest area at top); columns are 48 evenly-spaced slots
across the year (4 per month). Each cell shows the Sentinel-2 RGB
thumbnail of that paddock at the observation closest to the slot's
day-of-year, with non-paddock pixels masked black.

Useful for spotting cloud problems, cropping events, or stand-out
paddocks at a glance.

```python
from PaddockTS.Plotting.calendar_plot import calendar_plot

calendar_plot(q, thumb_size=64, max_paddocks_per_page=20)
# -> {out_dir}/<stem>_calendar_<year>_p01.png  (one per year × page)
```

::: PaddockTS.Plotting.calendar_plot.calendar_plot

### Phenology plot

Multi-panel PNG: rows are paddocks, columns are years. Each panel
overlays the raw vegetation-index series (filled blue dots) and the
resampled-and-smoothed series (open blue dots) on a DOY axis, with
SoS / PoS / EoS DOYs drawn as vertical reference lines.

```python
from PaddockTS.Plotting.phenology_plot import phenology_plot

phenology_plot(q, variable="NDVI", max_paddocks_per_page=8)
# -> {out_dir}/<stem>_phenology_p01.png
```

::: PaddockTS.Plotting.phenology_plot.phenology_plot

### OzWALD climate plot

One PNG per group (temperature, precipitation, wind, radiation).
Reads the cached daily CSV produced by `download_ozwald_daily`.

```python
from PaddockTS.Plotting.ozwald_plot import ozwald_daily_plot

ozwald_daily_plot(q)
# -> {out_dir}/<stub>_ozwald_daily_temperature.png
# -> {out_dir}/<stub>_ozwald_daily_precipitation.png
# -> {out_dir}/<stub>_ozwald_daily_wind.png
# -> {out_dir}/<stub>_ozwald_daily_radiation.png
```

::: PaddockTS.Plotting.ozwald_plot.ozwald_daily_plot

### SILO climate plot

One PNG per group (temperature, rainfall, radiation, evapotranspiration,
humidity). Reads the cached SILO CSV produced by `download_silo`.

```python
from PaddockTS.Plotting.silo_plot import silo_plot

silo_plot(q)
# -> {out_dir}/<stub>_silo_temperature.png  ... etc.
```

::: PaddockTS.Plotting.silo_plot.silo_plot

### Terrain plot

2 × 2 panel: elevation, D8 flow accumulation, aspect, slope. Reads
the Copernicus DEM downloaded by `download_terrain`, applies a Gaussian
smoother before flow analysis (sharp DEMs produce striped artefacts),
and reprojects to the Sentinel-2 grid for easy overlay.

```python
from PaddockTS.Plotting.terrain_tiles_plot import terrain_tiles_plot

terrain_tiles_plot(q, sigma=10)
# -> {out_dir}/<stub>_topography.png
```

::: PaddockTS.Plotting.terrain_tiles_plot.terrain_tiles_plot

---

## PDF report

`make_pdf` stitches every plot produced for a query into a single
A4-landscape PDF with section headers — Landscape (topography),
Climate (SILO, OzWALD), Satellite Calendar (SAM + user paddocks),
Phenology (SAM + user paddocks).

```python
from PaddockTS.Plotting.make_pdf import make_pdf

# SAM-only report
make_pdf(q)
# -> {out_dir}/<stub>.pdf

# Include user-paddocks sections too
make_pdf(q, paddocks_filepath="/path/to/paddocks.gpkg")
```

::: PaddockTS.Plotting.make_pdf.make_pdf
