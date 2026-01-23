#!/usr/bin/env python3
"""Visualize PaddockSegmentation2 output."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rioxarray
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from collections import Counter

# Paths
preseg_path = '/borevitz_projects/data/PaddockTSWeb/tmp/test_example_query/preseg.tif'
output_path = '/borevitz_projects/data/PaddockTSWeb/tmp/test_example_query/segmentation_viz.png'

# Load preseg GeoTIFF
preseg = rioxarray.open_rasterio(preseg_path)
print(f'Image shape: {preseg.shape}')

# Convert to (H, W, C) for visualization
image = preseg.values.transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
image_norm = image.astype(np.float32) / 255.0

# Run SLIC
segments = slic(image_norm, n_segments=50, compactness=10.0, channel_axis=-1, start_label=1)
n_segments = len(np.unique(segments))
print(f'SLIC produced {n_segments} segments')

# Calculate segment areas
segment_counts = Counter(segments.flatten())
pixel_area_ha = 10 * 10 / 10000  # 10m pixels = 0.01 ha
segment_areas = {k: v * pixel_area_ha for k, v in segment_counts.items()}
areas = list(segment_areas.values())
print(f'Segment area range: {min(areas):.2f} - {max(areas):.2f} ha')
print(f'Segments >= 10 ha: {sum(1 for a in areas if a >= 10)}')
print(f'Segments >= 1 ha: {sum(1 for a in areas if a >= 1)}')

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# 1. RGB composite of features
ax = axes[0, 0]
ax.imshow(image_norm)
ax.set_title('NDVI Temporal Features\n(R=Mean, G=Std, B=Range)')
ax.axis('off')

# 2. SLIC segments overlay
ax = axes[0, 1]
ax.imshow(mark_boundaries(image_norm, segments, color=(1, 0, 0), mode='thick'))
ax.set_title(f'SLIC Segments ({n_segments} segments)')
ax.axis('off')

# 3. Segment labels
ax = axes[1, 0]
im = ax.imshow(segments, cmap='tab20')
ax.set_title('Segment IDs')
ax.axis('off')
plt.colorbar(im, ax=ax, shrink=0.8)

# 4. Histogram of segment areas
ax = axes[1, 1]
ax.hist(areas, bins=20, edgecolor='black', alpha=0.7)
ax.axvline(x=10, color='red', linestyle='--', label='min_area_ha=10 (default)')
ax.axvline(x=1, color='green', linestyle='--', label='min_area_ha=1')
ax.set_xlabel('Segment Area (ha)')
ax.set_ylabel('Count')
ax.set_title('Segment Area Distribution')
ax.legend()

plt.suptitle('PaddockSegmentation2 Visualization\nTest Area: ~417 ha', fontsize=14)
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'\nSaved: {output_path}')
