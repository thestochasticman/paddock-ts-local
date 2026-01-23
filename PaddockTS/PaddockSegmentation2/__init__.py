"""
PaddockSegmentation2: scikit-image based paddock boundary detection.

This module provides an alternative to PaddockSegmentation that uses
classical computer vision (SLIC superpixels) instead of deep learning (SAM).

Advantages:
- No model downloads required
- Works offline
- Lighter weight dependencies (numpy, scipy, scikit-image)
- Faster on CPU for smaller images

Usage:
    from PaddockTS.PaddockSegmentation2.get_paddocks import get_paddocks
    from PaddockTS.query import Query

    query = Query(lat=-33.5, lon=148.4, buffer=0.05, ...)
    get_paddocks(query)

Pipeline stages:
    1. _1_presegment.py: Compute NDVI temporal features (mean, std, range)
    2. _2_segment.py: Run SLIC superpixels and filter polygons
"""

from PaddockTS.PaddockSegmentation2.get_paddocks import get_paddocks
from PaddockTS.PaddockSegmentation2._1_presegment import presegment
from PaddockTS.PaddockSegmentation2._2_segment import segment

__all__ = ['get_paddocks', 'presegment', 'segment']
