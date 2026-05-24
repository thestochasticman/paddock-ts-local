# Environmental data

Auxiliary downloads of climate, terrain, and soil data for the same
bounding box and date range as the Sentinel-2 pipeline. Each is a
small standalone function that writes a tidy file (CSV or GeoTIFF) to
`{query.tmp_dir}/Environmental/` (or to `query.terrain_path` for the
DEM, which is AOI-keyed since elevation is time-invariant).

| Source | What it provides | Auth required |
|---|---|---|
| **Copernicus DEM 30 m** | 30 m elevation, merged tiles, clipped to AOI | none |
| **OzWALD daily** | Daily meteorology (T, P, wind, radiation) at AOI centre | none |
| **SILO** | Daily climate (T, rain, radiation, ET, vapour pressure) at AOI centre | email |
| **SLGA** | 90 m soil texture / properties per depth, clipped to AOI | TERN API key |

The Sentinel-2 → PaddockTS chain itself doesn't depend on any of these
— they're independent context layers, useful for downstream analyses
that combine remote sensing with weather, soil, or topography.

---

## Terrain — Copernicus DEM 30 m

`download_terrain` builds the list of Copernicus DEM 30 m tile URLs
covering the AOI from
[`copernicus-dem-30m` on AWS](https://registry.opendata.aws/copernicus-dem/),
fetches each via `download_cogs` (which streams only the bytes that
intersect the AOI), and merges them into a single GeoTIFF written to
`query.terrain_path`.

Output is AOI-keyed (`{config.tmp_dir}/aoi/{bbox_hash}/terrain.tif`),
so queries with the same bbox but different date ranges share the
download.

### Example

```python
from datetime import date
from PaddockTS.query import Query
from PaddockTS.Environmental.TerrainTiles.download_terrain_tiles import download_terrain

q = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2024, 1, 1),
    end=date(2024, 12, 31),
    stub="env_demo",
)

path = download_terrain(q)
print(path)
# ~/Downloads/PaddockTS-Tmp/aoi/<bbox_hash>/terrain.tif
```

Use with [`terrain_tiles_plot`](plotting.md#terrain-plot) to render
elevation / slope / aspect / flow accumulation.

### Reference

::: PaddockTS.Environmental.TerrainTiles.download_terrain_tiles.download_terrain

---

## OzWALD daily climate

[OzWALD](https://www.wenfo.org/ozwald/) is the Australian Water and
Landscape Dynamics dataset hosted by ANU, providing modelled daily
meteorology over Australia at ~5 km resolution.

`download_ozwald_daily` opens the OPeNDAP-served per-year NetCDFs,
samples the nearest grid cell to the AOI centre, concatenates years,
and writes a tidy CSV (one row per day, one column per variable) to
`{tmp_dir}/Environmental/{stub}_ozwald_daily.csv`. Cached: if the CSV
already exists it is loaded and returned without contacting the
server.

### Example

```python
from PaddockTS.Environmental.OzWALD.download_ozwald_daily import download_ozwald_daily

df = download_ozwald_daily(q)
print(df.head())
#         time   Tmax   Tmin     Pg  ...   Ueff  DWLReff
# 0 2024-01-01  31.20  17.40   0.00  ...  3.85    345.7
# 1 2024-01-02  33.50  16.80   0.00  ...  3.12    349.2
# ...

# Fetch a custom subset
df = download_ozwald_daily(q, variables=["Tmax", "Tmin", "Pg"])
```

Variables include `Tmax`, `Tmin`, `Pg` (precipitation), `Uavg`,
`Ueff` (wind), and `DWLReff` (downwelling longwave radiation); the
full default set is read from the bundled `OzWALD` config.

### Reference

::: PaddockTS.Environmental.OzWALD.download_ozwald_daily.download_ozwald_daily

---

## OzWALD 8-day vegetation

`download_ozwald_8day` is the same shape as the daily downloader but
hits the 8-day vegetation aggregates (NDVI, EVI, PV, NPV, BS, LAI,
GPP, soil moisture, runoff).

### Example

```python
from PaddockTS.Environmental.OzWALD.download_ozwald_8day import download_ozwald_8day

df = download_ozwald_8day(q)
print(df.columns.tolist())
# ['time', 'NDVI', 'EVI', 'PV', 'NPV', 'BS', 'LAI', 'GPP', 'Ssoil', 'Qtot']
```

### Reference

::: PaddockTS.Environmental.OzWALD.download_ozwald_8day.download_ozwald_8day

---

## SILO climate

[SILO](https://www.longpaddock.qld.gov.au/silo/) is the Queensland
Government's daily climate database for Australia, with a continuous
record from 1889 onward via spatial interpolation of station data.

`download_silo` hits the SILO `DataDrillDataset` CGI endpoint at the
centre of `query.bbox`, requests every available variable, and writes
a tidy CSV.

**Requires** an email address registered with SILO via
`~/.config/PaddockTS.json` (`"email": "..."`) or passed explicitly:

```python
from PaddockTS.Environmental.SILO.download_silo import download_silo

df = download_silo(q, email="you@example.org")
# or, if email is set in ~/.config/PaddockTS.json:
df = download_silo(q)

print(df.columns.tolist())
# ['YYYY-MM-DD', 'daily_rain', 'max_temp', 'min_temp', 'radiation',
#  'vp', 'vp_deficit', 'et_short_crop', 'evap_pan', ...]
```

Cached: if the output CSV already exists it is loaded and returned
without contacting SILO.

### Reference

::: PaddockTS.Environmental.SILO.download_silo.download_silo

---

## SLGA soils

[SLGA](https://esoil.io/TERNLandscapes/Public/Pages/SLGA/index.html)
provides national-coverage 90 m grids of soil properties (clay, sand,
silt, organic carbon, pH, bulk density, etc.) at standard depths.

`download_slga_soils` fetches the cross-product of `vars × depths`,
clipped to `query.bbox`, with a matching quick-look PNG per layer.

**Requires** a TERN API key configured at `~/.config/PaddockTS.json`
(`"tern_api_key": "..."`). Generate one at
<https://account.tern.org.au/>.

### Example

```python
from PaddockTS.Environmental.SLGASoils.download_slgasoils import download_slga_soils

# Soil texture triple at the surface horizon
download_slga_soils(
    q,
    vars=["Clay", "Sand", "Silt"],
    depths=["5-15cm"],
)
# -> {tmp_dir}/Environmental/<stub>_Clay_5-15cm.tif   (+ quick-look PNG)
# -> {tmp_dir}/Environmental/<stub>_Sand_5-15cm.tif
# -> {tmp_dir}/Environmental/<stub>_Silt_5-15cm.tif

# Carbon profile through the soil
download_slga_soils(
    q,
    vars=["SOC"],
    depths=["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"],
)
```

### Reference

::: PaddockTS.Environmental.SLGASoils.download_slgasoils.download_slga_soils
