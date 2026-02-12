from xarray.core.dataset import Dataset
from sklearn.cluster import KMeans
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

def timeseries_kmeans(
    *indices: NDArray[np.float32],
    n_clusters: int = 8,
    verbose: bool = True,
) -> NDArray[np.int32]:
    h, w, t = indices[0].shape
    pixels = np.concatenate([idx.reshape(-1, t) for idx in indices], axis=1)
    nan_mask = np.any(np.isnan(pixels), axis=1)
    pixels = np.nan_to_num(pixels, nan=0.0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = np.full(h * w, -1, dtype=np.int32)
    labels[~nan_mask] = kmeans.fit_predict(pixels[~nan_mask])
    labels = labels.reshape(h, w)

    if verbose:
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f'  Cluster {u}: {c:,} pixels ({100*c/(h*w):.1f}%)')

    return labels


def compute_cluster_edges(labels: NDArray[np.int32]) -> NDArray[np.float32]:
    from scipy import ndimage
    h, w = labels.shape
    # Compute gradient of cluster labels
    # Where labels change, gradient is high
    sobel_x = ndimage.sobel(labels.astype(np.float32), axis=1)
    sobel_y = ndimage.sobel(labels.astype(np.float32), axis=0)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize to [0, 1]
    if edges.max() > 0:
        edges = edges / edges.max()

    # Apply gaussian smoothing to spread edges slightly
    edges = ndimage.gaussian_filter(edges, sigma=2)
    return edges.astype(np.float32)


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
    gdf['area_ha'] = gdf.geometry.area / 10000
    gdf['compactness'] = (4 * np.pi * gdf.geometry.area) / (gdf.geometry.length ** 2)
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


def find_optimal_clusters(
    *indices: NDArray[np.float32],
    transform,
    crs,
    k_range: range = range(2, 16),
    min_area_ha: float = 5,
    max_area_ha: float = 1500,
    min_compactness: float = 0.1,
    epsilon_factor: float = 0.005,
    method: str = 'contours',
    scoring: str = 'variance_ratio',
) -> dict:
    from sklearn.metrics import silhouette_score

    h, w, t = indices[0].shape
    pixels_flat = np.concatenate([idx.reshape(-1, t) for idx in indices], axis=1)
    pixels_flat = np.nan_to_num(pixels_flat, nan=0.0)

    results = []
    for k in k_range:
        labels = timeseries_kmeans(*indices, n_clusters=k, verbose=False)
        paddocks = labels_to_paddocks(labels, transform, crs, min_area_ha, max_area_ha, min_compactness, epsilon_factor, method)

        silhouette = 0.0
        if scoring in ('silhouette', 'combined'):
            labels_flat = labels.reshape(-1)
            n = len(labels_flat)
            if n > 10000:
                idx = np.random.default_rng(42).choice(n, 10000, replace=False)
                silhouette = silhouette_score(pixels_flat[idx], labels_flat[idx])
            else:
                silhouette = silhouette_score(pixels_flat, labels_flat)

        var_ratio = 0.0
        if scoring in ('variance_ratio', 'combined') and len(paddocks) > 0:
            metrics = evaluate_paddocks(paddocks, indices[0], transform)
            var_ratio = metrics['variance_ratio']

        coverage = paddocks.geometry.area.sum() / (h * w * abs(transform.a * transform.e)) if len(paddocks) > 0 else 0.0

        if scoring == 'silhouette':
            score = silhouette
        elif scoring == 'variance_ratio':
            score = var_ratio
        elif scoring == 'coverage':
            score = coverage
        elif scoring == 'combined':
            score = 0.5 * ((silhouette + 1) / 2) + 0.5 * min(var_ratio / 10, 1.0)
        else:
            score = var_ratio

        results.append({'k': k, 'n_paddocks': len(paddocks), 'silhouette': silhouette, 'variance_ratio': var_ratio, 'coverage': coverage, 'score': score})
        print(f'k={k}: {len(paddocks)} paddocks, silhouette={silhouette:.3f}, var_ratio={var_ratio:.2f}, coverage={coverage:.2f}, score={score:.3f}')

    optimal_idx = max(range(len(results)), key=lambda i: results[i]['score'])
    optimal_k = results[optimal_idx]['k']
    print(f'\nOptimal k={optimal_k} (score={results[optimal_idx]["score"]:.3f})')

    return {'optimal_k': optimal_k, 'results': results}

