# Sentinel-2 download

Downloads Sentinel-2 ARD tiles from a STAC catalog, applies cloud masking,
filters out scenes with too many missing pixels, and writes the result as
a Zarr dataset.

::: PaddockTS.Sentinel2.download_sentinel2
