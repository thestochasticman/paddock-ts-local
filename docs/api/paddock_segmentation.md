# Paddock segmentation

SAM-based segmentation pipeline that turns a multi-temporal Sentinel-2
stack into a `geopandas.GeoDataFrame` of paddock polygons.

Two stages internally:

1. **Presegmentation** — derives a single grayscale image from the
   Sentinel-2 stack using NDWI Fourier features, emphasising field
   boundaries.
2. **SAM mask generation** — feeds the presegmented image to Segment
   Anything (default: ViT-H, via `segment-geospatial`) and converts the
   resulting masks to vector polygons with area + compactness filtering.

::: PaddockTS.PaddockSegmentation.get_paddocks
