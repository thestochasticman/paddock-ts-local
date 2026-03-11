```
PaddockTS/
│
├── config.py                              # Global paths (out_dir, tmp_dir), loads ~/.config/PaddockTS.json
├── query.py                               # Query dataclass: bbox + dates → deterministic stub + paths
├── sentinel2_to_paddockTS_pipeline.py     # Orchestrator — runs all 13 steps with Rich progress table
├── run_environmental.py                   # Environmental-only pipeline (terrain, OzWALD, SILO, SLGA + plots)
├── get_outputs.py                         # Concurrent orchestrator — runs S2 pipeline + environmental in parallel threads
│
├── Sentinel2/                             # Step 1: Satellite imagery acquisition
│   └── download_sentinel2.py                  # STAC search → Dask load → cloud mask → zarr
│
├── IndicesAndVegFrac/                     # Steps 2–3: Spectral products
│   ├── indices.py                             # NDVI, CFI, NIRv, NDTI, CAI computation
│   └── veg_frac.py                            # Fractional cover via fractionalcover3 (bg, pv, npv)
│
├── PaddockSegmentation/                   # Step 5 (active): SAMGeo-based segmentation
│   └── get_paddocks.py                        # SAMGeo ViT-H → vectorise → filter by area/compactness
│
├── PaddockSegmentation2/                  # Alternative: K-Means + contour segmentation
│   └── get_paddocks.py                        # Full pipeline: preseg → cluster → filter → .gpkg
│
├── PaddockSegmentation3/                  # Experimental: W-Net (PyTorch) unsupervised segmentation
│   └── get_paddocks.py                        # Train W-Net → segment → filter → .gpkg
│
├── PaddockTS/                             # Steps 8–9: Per-paddock time-series extraction
│   ├── make_paddockTS.py                      # Rasterise paddocks, compute median per paddock per timestep
│   └── make_yearly_paddockTS.py               # Split by year, add day-of-year coordinate
│
├── Phenology/                             # Step 10: Phenological metric estimation
│   └── estimate_phenology.py                  # SoS, PoS, EoS via phenolopy per paddock per year
│
├── Plotting/                              # Steps 4, 6–7, 11–12 + reporting
│   ├── sentinel2_video.py                     # RGB composite → H.264 MP4 with date overlay
│   ├── vegfrac_video.py                       # bg/pv/npv → RGB video
│   ├── calendar_plot.py                       # Per-year paddock thumbnail calendar grids (PIL)
│   ├── phenology_plot.py                      # Scatter + interpolated NDVI with SoS/PoS/EoS markers
│   └── make_pdf.py                            # Assemble all plots into a single PDF report
│
└── Environmental/                         # Standalone environmental data acquisition
    ├── SILO/                                  # Australian Bureau of Meteorology point climate
    ├── OzWALD/                                # CSIRO satellite-derived land/water products
    ├── TerrainTiles/                          # Copernicus 30m DEM
    └── SLGASoils/                             # TERN Soil & Landscape Grid of Australia
```
