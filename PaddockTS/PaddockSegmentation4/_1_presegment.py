"""
Stage 1: Presegmentation using time series K-means clustering.

This module clusters pixels by their NDVI temporal signature,
grouping similar phenological patterns (crop types, grazing patterns).

Output: GeoTIFF with cluster labels + edge magnitude for watershed.
"""

import pickle
from os.path import exists

import numpy as np
import xarray as xr
import rioxarray
from numpy.typing import NDArray
from xarray.core.dataset import Dataset
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter

from PaddockTS.query import Query
from PaddockTS.PaddockSegmentation4.utils import completion, compute_cluster_edges, normalize


def compute_ndvi(ds: Dataset) -> NDArray[np.float32]:
    """Compute NDVI = (nir - red) / (nir + red) from Sentinel-2 dataset."""
    red = ds["nbart_red"].transpose("y", "x", "time").values.astype(np.float32)
    nir = ds["nbart_nir_1"].transpose("y", "x", "time").values.astype(np.float32)

    red[red == 0] = np.nan
    nir[nir == 0] = np.nan

    red /= 10000.0
    nir /= 10000.0

    den = nir + red
    ndvi = (nir - red) / den
    ndvi[~np.isfinite(ndvi)] = np.nan

    return ndvi


def timeseries_kmeans(
    ndvi: NDArray[np.float32],
    n_clusters: int = 8,
    verbose: bool = True,
) -> NDArray[np.int32]:
    """
    Cluster pixels by their NDVI time series.

    Each pixel's temporal profile becomes its feature vector.
    Pixels with similar phenological patterns cluster together.

    Args:
        ndvi: NDVI time series (H, W, T)
        n_clusters: Number of land cover clusters
        verbose: Print cluster info

    Returns:
        Cluster labels (H, W) with values 0 to n_clusters-1
    """
    h, w, t = ndvi.shape

    # Flatten to (n_pixels, n_timesteps)
    pixels = ndvi.reshape(-1, t)

    # Handle NaN - replace with 0
    nan_mask = np.any(np.isnan(pixels), axis=1)
    pixels = np.nan_to_num(pixels, nan=0.0)

    # Run K-means on time series
    if verbose:
        print(f"Clustering {h*w} pixels by {t}-step time series into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)

    # Reshape back to image
    labels = labels.reshape(h, w).astype(np.int32)

    # Print cluster sizes
    if verbose:
        unique, counts = np.unique(labels, return_counts=True)
        print("Cluster sizes (pixels):")
        for u, c in zip(unique, counts):
            print(f"  Cluster {u}: {c:,} pixels ({100*c/(h*w):.1f}%)")

    return labels


def find_optimal_clusters(
    ndvi: NDArray[np.float32],
    transform,
    crs,
    k_range: range = range(2, 16),
    min_area_ha: float = 5,
    max_area_ha: float = 1500,
    min_compactness: float = 0.1,
    marker_percentile: float = 25,
    scoring: str = 'coverage',
) -> dict:
    """
    Find optimal number of clusters using specified scoring method.

    For each k, runs the full pipeline and measures output quality.

    Args:
        ndvi: Gap-filled NDVI time series (H, W, T)
        transform: Affine transform for georeferencing
        crs: Coordinate reference system
        k_range: Range of k values to test (default 2-15)
        min_area_ha: Minimum paddock area for filtering
        max_area_ha: Maximum paddock area for filtering
        min_compactness: Minimum compactness for filtering
        marker_percentile: Percentile for watershed markers
        scoring: Scoring method - 'coverage', 'silhouette', or 'combined'

    Returns:
        dict with 'optimal_k', 'results' (metrics per k), 'total_area_ha'
    """
    from scipy import ndimage
    from skimage import segmentation, morphology
    from rasterio import features
    from shapely.geometry import shape
    import geopandas as gpd

    h, w, t = ndvi.shape
    pixel_area = abs(transform.a * transform.e)
    total_area_ha = (h * w * pixel_area) / 10000

    # Flatten pixels for silhouette scoring
    pixels_flat = ndvi.reshape(-1, t)
    pixels_flat = np.nan_to_num(pixels_flat, nan=0.0)

    print(f"\n--- Finding optimal clusters (k={k_range.start}-{k_range.stop-1}, scoring={scoring}) ---")

    results = []

    for k in k_range:
        # Run clustering
        labels = timeseries_kmeans(ndvi, n_clusters=k, verbose=False)
        labels_flat = labels.reshape(-1)

        # Compute silhouette score (subsample for speed if large)
        silhouette = 0.0
        if scoring in ('silhouette', 'combined'):
            from sklearn.metrics import silhouette_score
            n_samples = len(labels_flat)
            if n_samples > 10000:
                # Subsample for speed
                idx = np.random.choice(n_samples, 10000, replace=False)
                silhouette = silhouette_score(pixels_flat[idx], labels_flat[idx])
            else:
                silhouette = silhouette_score(pixels_flat, labels_flat)

        # Compute cluster edges
        edges = compute_cluster_edges(labels)

        # Run watershed on edges
        edges_norm = edges / (edges.max() + 1e-8)
        threshold = np.percentile(edges_norm, marker_percentile)
        markers = edges_norm <= threshold
        markers = morphology.remove_small_objects(markers, min_size=50)
        markers = morphology.remove_small_holes(markers, area_threshold=25)
        markers = morphology.binary_erosion(markers, morphology.disk(3))
        markers_labeled, n_markers = ndimage.label(markers)

        if n_markers == 0:
            results.append({
                'k': k, 'n_paddocks': 0, 'coverage_pct': 0,
                'silhouette': silhouette, 'score': 0,
            })
            print(f"k={k}: 0 paddocks, 0.0% coverage, silhouette={silhouette:.3f}")
            continue

        segments = segmentation.watershed(edges_norm, markers_labeled)

        # Convert to polygons
        shapes_gen = features.shapes(segments.astype(np.int32), transform=transform)
        records = []
        for geom, value in shapes_gen:
            if value == 0:
                continue
            records.append({'geometry': shape(geom), 'segment_id': int(value)})

        if not records:
            results.append({
                'k': k, 'n_paddocks': 0, 'coverage_pct': 0,
                'silhouette': silhouette, 'score': 0,
            })
            print(f"k={k}: 0 paddocks, 0.0% coverage, silhouette={silhouette:.3f}")
            continue

        gdf = gpd.GeoDataFrame(records, crs=crs)
        gdf['area_ha'] = gdf.geometry.area / 10000
        gdf['perimeter'] = gdf.geometry.length
        gdf['compactness'] = (4 * np.pi * gdf.geometry.area) / (gdf['perimeter'] ** 2)

        # Filter
        mask = (
            (gdf['area_ha'] >= min_area_ha) &
            (gdf['area_ha'] <= max_area_ha) &
            (gdf['compactness'] >= min_compactness)
        )
        gdf_filtered = gdf[mask]

        n_paddocks = len(gdf_filtered)
        paddock_area = gdf_filtered['area_ha'].sum() if n_paddocks > 0 else 0
        coverage_pct = (paddock_area / total_area_ha) * 100

        # Compute score based on method
        if scoring == 'coverage':
            score = coverage_pct
        elif scoring == 'silhouette':
            score = silhouette
        elif scoring == 'combined':
            # Normalize coverage to 0-1 and combine with silhouette
            coverage_norm = coverage_pct / 100
            # Silhouette is already -1 to 1, shift to 0-1
            silhouette_norm = (silhouette + 1) / 2
            score = 0.5 * coverage_norm + 0.5 * silhouette_norm
        else:
            score = coverage_pct

        results.append({
            'k': k,
            'n_paddocks': n_paddocks,
            'paddock_area_ha': paddock_area,
            'coverage_pct': coverage_pct,
            'silhouette': silhouette,
            'score': score,
        })

        print(f"k={k}: {n_paddocks} paddocks, {coverage_pct:.1f}% coverage, "
              f"silhouette={silhouette:.3f}, score={score:.3f}")

    # Find optimal k (maximize score)
    optimal_idx = max(range(len(results)), key=lambda i: results[i]['score'])
    optimal_k = results[optimal_idx]['k']

    print(f"\nOptimal k={optimal_k} (score={results[optimal_idx]['score']:.3f})")

    return {
        'optimal_k': optimal_k,
        'results': results,
        'total_area_ha': total_area_ha,
    }


def compute_preseg_features(
    ds: Dataset,
    n_clusters: int | str = 8,
    min_area_ha: float = 5,
    max_area_ha: float = 1500,
    min_compactness: float = 0.1,
    k_range: range = range(4, 16),
    scoring: str = 'coverage',
) -> NDArray[np.float32]:
    """
    Compute time series K-means labels and cluster edges.

    Args:
        ds: xarray Dataset with Sentinel-2 bands
        n_clusters: Number of clusters for K-means, or 'auto' to find optimal
        min_area_ha: Min paddock area (used when n_clusters='auto')
        max_area_ha: Max paddock area (used when n_clusters='auto')
        min_compactness: Min compactness (used when n_clusters='auto')
        k_range: Range of k values to try (used when n_clusters='auto')
        scoring: Scoring method - 'coverage', 'silhouette', or 'combined'

    Returns:
        Feature array (H, W, 3):
        - Band 0: Normalized cluster labels
        - Band 1: Cluster edge magnitude
        - Band 2: Median NDVI (for visualization)
    """
    # Compute and gap-fill NDVI
    ndvi = compute_ndvi(ds)
    ndvi_filled = completion(ndvi)

    # Apply Gaussian smoothing to each timestep
    sigma = 1.0
    for t in range(ndvi_filled.shape[2]):
        ndvi_filled[:, :, t] = gaussian_filter(ndvi_filled[:, :, t], sigma=sigma)

    # Find optimal k if auto
    if n_clusters == 'auto':
        # Get transform and CRS for evaluation
        transform = ds.rio.transform()
        crs = ds.rio.crs

        opt_result = find_optimal_clusters(
            ndvi_filled, transform, crs,
            k_range=k_range,
            min_area_ha=min_area_ha,
            max_area_ha=max_area_ha,
            min_compactness=min_compactness,
            scoring=scoring,
        )
        n_clusters = opt_result['optimal_k']

    # Run time series K-means
    labels = timeseries_kmeans(ndvi_filled, n_clusters=n_clusters)

    # Compute edges at cluster boundaries
    edges = compute_cluster_edges(labels)

    # Compute median NDVI for visualization (3rd band)
    ndvi_median = np.nanmedian(ndvi_filled, axis=2)

    # Normalize all to [0, 1] for uint8 storage
    labels_norm = normalize(labels.astype(np.float32))
    ndvi_median_norm = normalize(ndvi_median)

    # Stack features: labels, edges, median NDVI (3 bands for RGB plotting)
    features = np.stack([labels_norm, edges, ndvi_median_norm], axis=-1)

    return features.astype(np.float32)


def rescale_uint8(im: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Scale each band independently to [0, 255] as uint8."""
    if im.ndim == 2:
        im = im[:, :, None]

    h, w, b = im.shape
    out = np.empty((h, w, b), dtype=np.uint8)

    for i in range(b):
        band = im[:, :, i]
        finite = np.isfinite(band)

        if not np.any(finite):
            out[:, :, i] = 0
            continue

        vmin = np.nanmin(band)
        vmax = np.nanmax(band)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            out[:, :, i] = 0
            continue

        scaled = (band - vmin) / (vmax - vmin)
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled[~finite] = 0.0
        out[:, :, i] = (scaled * 255.0).astype(np.uint8)

    return out


def convert_to_geotiff(ds2: Dataset, inp_uint8: NDArray[np.uint8]) -> xr.DataArray:
    """Wrap uint8 H x W x B array into a georeferenced DataArray."""
    if inp_uint8.ndim == 2:
        inp_uint8 = inp_uint8[:, :, None]

    image = inp_uint8
    lat = ds2.y.values
    lon = ds2.x.values
    bands = np.arange(1, image.shape[2] + 1)

    data_xr = xr.DataArray(
        image,
        coords={"y": lat, "x": lon, "band": bands},
        dims=("y", "x", "band"),
    )

    data_xr.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    if ds2.rio.crs is not None:
        data_xr.rio.write_crs(ds2.rio.crs, inplace=True)
    try:
        data_xr.rio.write_transform(ds2.rio.transform(), inplace=True)
    except Exception:
        pass

    return data_xr


def presegment(
    query: Query,
    n_clusters: int | str = 'auto',
    min_area_ha: float = 5,
    max_area_ha: float = 1500,
    min_compactness: float = 0.1,
    k_range: range = range(4, 16),
    scoring: str = 'coverage',
) -> xr.DataArray:
    """
    Main entry point for Stage 1 presegmentation.

    Runs time series K-means and saves cluster labels + edges as GeoTIFF.

    Args:
        query: Query object with paths
        n_clusters: Number of K-means clusters, or 'auto' to find optimal (default)
        min_area_ha: Min paddock area (used when n_clusters='auto')
        max_area_ha: Max paddock area (used when n_clusters='auto')
        min_compactness: Min compactness (used when n_clusters='auto')
        k_range: Range of k values to try (used when n_clusters='auto')
        scoring: Scoring method - 'coverage', 'silhouette', or 'combined'

    Returns:
        Georeferenced DataArray with preseg features
    """
    if not exists(query.path_ds2):
        raise FileNotFoundError(
            f"Sentinel-2 data not found at {query.path_ds2}. "
            "Run download_sentinel2(query) first."
        )

    ds2 = pickle.load(open(query.path_ds2, 'rb'))

    print(f"\n=== PaddockSegmentation3: Time Series K-means ===")
    print(f"Time steps: {len(ds2.time)}")

    features = compute_preseg_features(
        ds2,
        n_clusters=n_clusters,
        min_area_ha=min_area_ha,
        max_area_ha=max_area_ha,
        min_compactness=min_compactness,
        k_range=k_range,
        scoring=scoring,
    )
    u8 = rescale_uint8(features)
    preseg_geotiff = convert_to_geotiff(ds2, u8)

    # Save
    preseg_geotiff.transpose("band", "y", "x").rio.to_raster(query.path_preseg_tif)
    print(f"Saved preseg to {query.path_preseg_tif}")

    return preseg_geotiff


def test():
    from PaddockTS.query import get_example_query
    from os import remove

    query = get_example_query()
    if exists(query.path_preseg_tif):
        remove(query.path_preseg_tif)
    presegment(query)
    print(f"Output: {query.path_preseg_tif}")
    return exists(query.path_preseg_tif)


if __name__ == '__main__':
    print(test())
