import numpy as np
import xarray as xr
from os.path import exists
from PaddockTS.query import Query
from .preprocess import preprocess
from .utils import compute_ndvi, compute_ndwi, labels_to_paddocks, evaluate_paddocks


def get_paddocks(
    query: Query,
    clustering: str = 'kmeans',
    n_clusters: int | str = 'auto',
    min_area_ha: float = 5,
    max_area_ha: float = 1500,
    min_compactness: float = 0.0,
    k_range: range = range(3, 16),
    scoring: str = 'silhouette',
    epsilon_factor: float = 0.005,
    method: str = 'contours',
    n_classes: int = 5,
    max_epochs: int = 500,
    ncut_weight: float = 1.0,
):
    from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2

    if not exists(query.sentinel2_path):
        download_sentinel2(query)

    ds = xr.open_zarr(query.sentinel2_path)
    ndvi = compute_ndvi(ds)

    if clustering == 'wnet':
        from PaddockTS.PaddockSegmentation3.wnet import segment_wnet
        ndwi = compute_ndwi(ds)
        labels = segment_wnet(ndvi, ndwi, n_classes=n_classes, max_epochs=max_epochs, ncut_weight=ncut_weight)
    else:
        geotiff = preprocess(query, n_clusters=n_clusters, min_area_ha=min_area_ha,
                             max_area_ha=max_area_ha, min_compactness=min_compactness,
                             k_range=k_range, scoring=scoring,
                             epsilon_factor=epsilon_factor, method=method)
        labels = geotiff[:, :, 0].values

    paddocks = labels_to_paddocks(labels, transform=ds.rio.transform(), crs=ds.rio.crs,
                                  min_area_ha=min_area_ha, max_area_ha=max_area_ha,
                                  min_compactness=min_compactness, epsilon_factor=epsilon_factor, method=method)

    metrics = evaluate_paddocks(paddocks, ndvi, ds.rio.transform())
    print(f'{metrics["n_paddocks"]} paddocks | '
          f'within var: {metrics["mean_within_variance"]:.4f} | '
          f'between var: {metrics["between_variance"]:.4f} | '
          f'ratio: {metrics["variance_ratio"]:.2f}')

    gpkg_path = f'{query.tmp_dir}/{query.stub}_paddocks.gpkg'
    paddocks.to_file(gpkg_path, driver='GPKG')
    print(f'Saved to {gpkg_path}')
    return paddocks, labels


def test():
    from PaddockTS.utils import get_example_query
    import matplotlib.pyplot as plt
    import rioxarray

    query = get_example_query()
    paddocks, labels = get_paddocks(query, scoring='variance_ratio', method='rasterio')

    ds = xr.open_zarr(query.sentinel2_path)
    ndvi_median = np.nanmedian(compute_ndvi(ds), axis=2)
    x, y = ds.x.values, ds.y.values
    extent = [x.min(), x.max(), y.min(), y.max()]

    preseg = rioxarray.open_rasterio(f'{query.tmp_dir}/{query.stub}_preseg.tif')
    titles = ['Clusters', 'Edges', 'Median NDVI', 'Median NDWI', f'{len(paddocks)} Paddocks']
    cmaps = ['tab10', 'gray', 'RdYlGn', 'RdYlBu', 'RdYlGn']

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for i in range(4):
        axes[i].imshow(preseg[i].values, cmap=cmaps[i], extent=extent, origin='upper')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    axes[4].imshow(ndvi_median, cmap='RdYlGn', extent=extent, origin='upper')
    paddocks.boundary.plot(ax=axes[4], color='red', linewidth=2)
    axes[4].set_title(titles[4])
    axes[4].axis('off')

    png_path = f'{query.tmp_dir}/{query.stub}_paddocks.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f'Saved plot to {png_path}')

    # per-paddock plots
    import geopandas as gpd
    n = len(paddocks)
    fig2, axes2 = plt.subplots(n, 5, figsize=(25, 5 * n))
    if n == 1:
        axes2 = axes2[np.newaxis, :]
    for row, (_, p) in enumerate(paddocks.iterrows()):
        minx, miny, maxx, maxy = p.geometry.bounds
        pad = max(maxx - minx, maxy - miny) * 0.2
        xlim, ylim = (minx - pad, maxx + pad), (miny - pad, maxy + pad)
        for col in range(4):
            axes2[row, col].imshow(preseg[col].values, cmap=cmaps[col], extent=extent, origin='upper')
            axes2[row, col].set_xlim(xlim)
            axes2[row, col].set_ylim(ylim)
            if row == 0:
                axes2[row, col].set_title(titles[col])
            axes2[row, col].axis('off')
        axes2[row, 4].imshow(ndvi_median, cmap='RdYlGn', extent=extent, origin='upper')
        gpd.GeoSeries([p.geometry], crs=paddocks.crs).boundary.plot(ax=axes2[row, 4], color='red', linewidth=2)
        axes2[row, 4].set_xlim(xlim)
        axes2[row, 4].set_ylim(ylim)
        axes2[row, 4].set_ylabel(f'{p.area_ha:.0f} ha', rotation=0, labelpad=40)
        if row == 0:
            axes2[row, 4].set_title('Paddock')
        axes2[row, 4].axis('off')

    plt.tight_layout()
    per_path = f'{query.tmp_dir}/{query.stub}_per_paddock.png'
    plt.savefig(per_path, dpi=150, bbox_inches='tight')
    print(f'Saved per-paddock plot to {per_path}')
    plt.show()

if __name__ == '__main__':
    test()
