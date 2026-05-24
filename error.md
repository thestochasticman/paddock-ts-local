```
(paddockts) adeel@yasar-m1-macbook-air paddock-ts-local % python PaddockTS/get_outputs.py 
                              Environmental                                                             Sentinel-2 → PaddockTS                          
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓    ┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ #   ┃ Step                                 ┃ Status       ┃ Time       ┃    ┃ #   ┃ Step                                 ┃ Status       ┃ Time       ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩    ┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 1   │ Download terrain                     │ done         │ 0.4s       │    │ 1   │ Download Sentinel-2                  │ done         │ 0.5s       │
│ 2   │ Download OzWALD daily                │ done         │ 0.0s       │    │ 2   │ Compute indices                      │ done         │ 0.1s       │
│ 3   │ Download SILO                        │ done         │ 0.0s       │    │ 3   │ Compute fractional cover             │ done         │ 11.8s      │
│ 4   │ Download SLGA soils                  │ error        │ 2.5s       │    │ 4   │ Sentinel-2 video                     │ done         │ 1.5s       │
│ 5   │ OzWALD plot                          │ done         │ 0.7s       │    │ 5   │ Segment paddocks (SAM)               │ done         │ 0.0s       │
│ 6   │ SILO plot                            │ done         │ 0.9s       │    │ 6   │ S2 + paddocks video (SAM)            │ done         │ 1.5s       │
│ 7   │ Terrain plot                         │ done         │ 8.3s       │    │ 7   │ S2 + paddocks video (user)           │ done         │ 1.3s       │
└─────┴──────────────────────────────────────┴──────────────┴────────────┘    │ 8   │ Fractional cover video               │ done         │ 1.3s       │
                                                                              │ 9   │ FC + paddocks video (SAM)            │ done         │ 1.5s       │
                                                                              │ 10  │ FC + paddocks video (user)           │ done         │ 1.4s       │
                                                                              │ 11  │ Make paddock TS (SAM)                │ done         │ 3.0s       │
                                                                              │ 12  │ Make paddock TS (user)               │ done         │ 2.4s       │
                                                                              │ 13  │ Make yearly paddock TS (SAM)         │ done         │ 0.2s       │
                                                                              │ 14  │ Make yearly paddock TS (user)        │ done         │ 0.1s       │
                                                                              │ 15  │ Estimate phenology (SAM)             │ done         │ 0.6s       │
                                                                              │ 16  │ Estimate phenology (user)            │ done         │ 0.0s       │
                                                                              │ 17  │ Calendar plot (SAM)                  │ done         │ 0.5s       │
                                                                              │ 18  │ Calendar plot (user)                 │ done         │ 0.5s       │
                                                                              │ 19  │ Phenology plot (SAM)                 │ done         │ 1.1s       │
                                                                              │ 20  │ Phenology plot (user)                │ done         │ 0.9s       │
                                                                              │ 21  │ Make PDF report                      │ done         │ 2.0s       │
                                                                              └─────┴──────────────────────────────────────┴──────────────┴────────────┘
╭──────────────────────────────────────────────────────────────────────────────────────────── log ────────────────────────────────────────────────────────────────────────────────────────────╮
│ Phenometrics calculated successfully!                                                                                                                                                       │
│   2024: 11 paddocks, 2.1 avg peaks                                                                                                                                                          │
│   2025: skipped (no paddocks with >= 25 observations)                                                                                                                                       │
│ Saved to /Users/adeel/borevitz_projects/data/PaddockTSWeb/PaddockSet1/sam_paddocks_calendar_2024_p01.png                                                                                    │
│ Saved to /Users/adeel/borevitz_projects/data/PaddockTSWeb/PaddockSet1/sam_paddocks_calendar_2025_p01.png                                                                                    │
│ Saved to /Users/adeel/borevitz_projects/data/PaddockTSWeb/PaddockSet1/PaddockSet1_calendar_2024_p01.png                                                                                     │
│ Saved to /Users/adeel/borevitz_projects/data/PaddockTSWeb/PaddockSet1/PaddockSet1_calendar_2025_p01.png                                                                                     │
│ Saved to /Users/adeel/borevitz_projects/data/PaddockTSWeb/PaddockSet1/sam_paddocks_phenology_p01.png                                                                                        │
│ Saved to /Users/adeel/borevitz_projects/data/PaddockTSWeb/PaddockSet1/sam_paddocks_phenology_p02.png                                                                                        │
│ Saved to /Users/adeel/borevitz_projects/data/PaddockTSWeb/PaddockSet1/PaddockSet1_phenology_p01.png                                                                                         │
│ Saved to /Users/adeel/borevitz_projects/data/PaddockTSWeb/PaddockSet1/PaddockSet1_phenology_p02.png                                                                                         │
│ PDF report saved to /Users/adeel/borevitz_projects/data/PaddockTSWeb/PaddockSet1/PaddockSet1_report.pdf                                                                                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Environmental: FAILED — Failed to access COG for Clay 5-15cm: HTTP response code: 404
Traceback (most recent call last):
  File "rasterio/_base.pyx", line 311, in rasterio._base.DatasetBase.__init__
  File "rasterio/_base.pyx", line 222, in rasterio._base.open_dataset
  File "rasterio/_err.pyx", line 359, in rasterio._err.exc_wrap_pointer
rasterio._err.CPLE_HttpResponseError: HTTP response code: 404

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/adeel/borevitz_projects/repos/paddock-ts-local/PaddockTS/Environmental/SLGASoils/download_cog.py", line 21, in download_cog
    with rasterio.open(f'/vsicurl/{url}') as src:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adeel/miniconda3/envs/paddockts/lib/python3.11/site-packages/rasterio/env.py", line 463, in wrapper
    return f(*args, **kwds)
           ^^^^^^^^^^^^^^^^
  File "/Users/adeel/miniconda3/envs/paddockts/lib/python3.11/site-packages/rasterio/__init__.py", line 356, in open
    dataset = DatasetReader(path, driver=driver, sharing=sharing, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "rasterio/_base.pyx", line 313, in rasterio._base.DatasetBase.__init__
rasterio.errors.RasterioIOError: HTTP response code: 404

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/adeel/borevitz_projects/repos/paddock-ts-local/PaddockTS/get_outputs.py", line 687, in <module>
    get_outputs(query, reload='--reload' in sys.argv, paddocks_filepath=fp, label_col='paddock', show_log=True)
  File "/Users/adeel/borevitz_projects/repos/paddock-ts-local/PaddockTS/get_outputs.py", line 679, in get_outputs
    raise errors[0][1]
  File "/Users/adeel/borevitz_projects/repos/paddock-ts-local/PaddockTS/get_outputs.py", line 639, in env_worker
    _run_env_steps(query, env_statuses, env_times, errors=errors)
  File "/Users/adeel/borevitz_projects/repos/paddock-ts-local/PaddockTS/get_outputs.py", line 287, in _run_env_steps
    raise step_errors[0][1]
  File "/Users/adeel/borevitz_projects/repos/paddock-ts-local/PaddockTS/get_outputs.py", line 253, in _run_env_steps
    download_slga_soils(query)
  File "/Users/adeel/borevitz_projects/repos/paddock-ts-local/PaddockTS/Environmental/SLGASoils/download_slgasoils.py", line 43, in download_slga_soils
    list(starmap(download_cog, args))
  File "/Users/adeel/borevitz_projects/repos/paddock-ts-local/PaddockTS/Environmental/SLGASoils/download_cog.py", line 34, in download_cog
    raise RuntimeError(f'Failed to access COG for {attribute} {depth}: {e}')    
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Failed to access COG for Clay 5-15cm: HTTP response code: 404
```
