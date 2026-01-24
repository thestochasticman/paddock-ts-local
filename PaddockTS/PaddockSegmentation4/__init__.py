"""
PaddockSegmentation4: OpenCV contour-based segmentation.

Key improvement over PaddockSegmentation3:
- Uses cv2.findContours to extract polygons directly from cluster labels
- No watershed needed - simpler and cleaner boundaries
- cv2.approxPolyDP simplifies jagged edges
"""

from PaddockTS.PaddockSegmentation4.get_paddocks import get_paddocks

__all__ = ['get_paddocks']
