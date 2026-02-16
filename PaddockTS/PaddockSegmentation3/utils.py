from xarray.core.dataset import Dataset
from numpy.typing import NDArray
import numpy as np

def _band(ds: Dataset, name: str) -> NDArray[np.float32]:
    b = ds[name].transpose('y', 'x', 'time').values.astype(np.float32)
    b[b == 0] = np.nan
    b /= 10000.0
    return b

def _normalised_diff(a, b):
    nd = (a - b) / (a + b)
    nd[~np.isfinite(nd)] = np.nan
    return nd

def compute_ndvi(ds: Dataset) -> NDArray[np.float32]:
    return _normalised_diff(_band(ds, 'nbart_nir_1'), _band(ds, 'nbart_red'))

def compute_ndwi(ds: Dataset) -> NDArray[np.float32]:
    return _normalised_diff(_band(ds, 'nbart_green'), _band(ds, 'nbart_nir_1'))

def _contours_to_records(labels, transform, epsilon_factor):
    import cv2
    from shapely.geometry import Polygon

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    records = []
    for label in np.unique(labels):
        if label == -1:
            continue
        mask = (labels == label).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 3:
                continue
            cnt = cv2.approxPolyDP(cnt, epsilon_factor * cv2.arcLength(cnt, True), True)
            coords = [(transform.c + pt[0] * transform.a, transform.f + pt[1] * transform.e) for pt in cnt.reshape(-1, 2)]
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            try:
                poly = Polygon(coords)
                if poly.is_valid and poly.area > 0:
                    records.append({'geometry': poly, 'cluster': int(label)})
            except Exception:
                continue
    return records

def _rasterio_to_records(labels, transform):
    from rasterio.features import shapes
    from shapely.geometry import shape
    return [{'geometry': shape(geom), 'cluster': int(val)}
            for geom, val in shapes(labels.astype(np.int32), connectivity=8, transform=transform) if val != -1]

def labels_to_paddocks(labels, transform, crs, min_area_ha=5, max_area_ha=1500, min_compactness=0.1, epsilon_factor=0.005, method='contours'):
    import geopandas as gpd

    if method == 'contours':
        records = _contours_to_records(labels, transform, epsilon_factor)
    else:
        records = _rasterio_to_records(labels, transform)

    if not records:
        return gpd.GeoDataFrame(columns=['geometry', 'cluster', 'area_ha', 'compactness'], crs=crs)

    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf.geometry = gdf.geometry.make_valid()
    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    metric = gdf.to_crs(gdf.estimate_utm_crs())
    gdf['area_ha'] = metric.geometry.area / 10000
    gdf['compactness'] = (4 * np.pi * metric.geometry.area) / (metric.geometry.length ** 2)
    return gdf[(gdf['area_ha'] >= min_area_ha) & (gdf['area_ha'] <= max_area_ha) & (gdf['compactness'] >= min_compactness)]


def evaluate_paddocks(paddocks, ndvi, transform):
    from rasterio.features import rasterize
    h, w = ndvi.shape[:2]
    ndvi_median = np.nanmedian(ndvi, axis=2)

    shapes = [(geom, i + 1) for i, geom in enumerate(paddocks.geometry)]
    paddock_raster = rasterize(shapes, out_shape=(h, w), transform=transform, fill=0, dtype=np.int32)

    counts, variances, means = [], [], []
    for pid in range(1, len(paddocks) + 1):
        vals = ndvi_median[paddock_raster == pid]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 1:
            counts.append(len(vals))
            variances.append(np.var(vals))
            means.append(np.mean(vals))

    if len(counts) < 2:
        return {'mean_within_variance': np.nan, 'between_variance': np.nan, 'variance_ratio': np.nan, 'n_paddocks': len(paddocks)}

    n = np.array(counts, dtype=np.float64)
    N = n.sum()
    within = np.sum(n * np.array(variances)) / N
    global_mean = np.sum(n * np.array(means)) / N
    between = np.sum(n * (np.array(means) - global_mean) ** 2) / N
    ratio = between / (within + 1e-8)

    return {
        'mean_within_variance': within,
        'between_variance': between,
        'variance_ratio': ratio,
        'n_paddocks': len(paddocks),
    }
