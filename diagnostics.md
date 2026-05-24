# Sentinel-2 download — known failure modes

## 1. `pystac_client.exceptions.APIError: 504 Gateway Time-out`

**Symptom.** First call to `pystac_client.Client.open('https://explorer.dea.ga.gov.au/stac')` (or the subsequent `catalog.search(...).items()`) raises a 504. Re-running the same script seconds later often succeeds.

**Diagnosis.** DEA's STAC fronting load-balancer takes ~30s on a cold request, then warms up. Reproducible from the shell:

```sh
$ for i in 1 2 3 4 5; do
    curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" --max-time 30 https://explorer.dea.ga.gov.au/stac
  done
000 30.001406s
200 1.792604s
200 0.140180s
200 0.136704s
200 0.171495s
```

The DEA status page (https://status.dea.ga.gov.au) reports "operational" while this is happening — the slow first request doesn't trip their health checks.

**Fix.** `download_sentinel2` now passes a `StacApiIO` with `urllib3.Retry` (5 retries, exponential backoff, on 408/429/502/503/504) to `pystac_client.Client.open(...)`. Backoff delays are 0s, 1s, 2s, 4s, 8s — covers the ~30s cold-start window without hammering.

## 2. `RasterioIOError('Unsupported Authorization Type')` from worker

**Symptom.** STAC search succeeds, but during the `odc.stac.load(...)` Dask compute, individual asset reads fail in a worker with `RasterioIOError('Unsupported Authorization Type')`.

**Diagnosis.** GDAL/CURL message that fires when GDAL is asked to use an HTTP auth scheme it doesn't recognise. Not caused by env var leakage — verified with `env | grep -iE '^(AWS_|GDAL_|CPL_|VSI|GS_|PROJ_|GS_OAUTH|AZURE_|CURL_)'` returning empty.

Most likely cause: `odc.stac.configure_rio(..., client=client)` broadcasts the unsigned-AWS GDAL config to dask workers via a plugin, but on newer `dask>=2026` the broadcast can race the first task — the worker opens the asset with default GDAL settings, tries some HTTP auth scheme that DEA's public S3 bucket doesn't support, and CURL returns this error.

**Mitigation candidates** (not yet committed in `download_sentinel2`):

- Set GDAL/AWS config in main-process env *before* spawning the Dask client so worker subprocesses inherit it at startup:
  ```python
  os.environ.setdefault('AWS_NO_SIGN_REQUEST', 'YES')
  os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
  os.environ.setdefault('CPL_VSIL_CURL_USE_HEAD', 'NO')
  ```
- Drop the distributed `DaskClient` and compute with the threaded scheduler — workers run in the same process, no broadcast needed:
  ```python
  ds = ds.compute(scheduler='threads')
  ```

The intermittent nature suggests it co-occurs with whatever causes the 504 above; if DEA-side flakiness is the upstream trigger, the retry fix in §1 may reduce frequency here too.

## Reproduction

Run the test suite:
```sh
python -m PaddockTS.Sentinel2.download_sentinel2
```

If the cold-cache 504 reappears, confirm with the `curl` loop above before assuming a regression in this module.
