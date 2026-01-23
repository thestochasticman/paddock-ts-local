"""
PaddockSegmentation3: Time series K-means + Watershed segmentation.

This approach clusters pixels by their temporal signature (phenology),
then uses watershed to split clusters into spatially connected paddocks.
"""

from PaddockTS.PaddockSegmentation3.get_paddocks import get_paddocks

__all__ = ['get_paddocks']
