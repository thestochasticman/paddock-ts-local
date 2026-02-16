import numpy as np
import xarray as xr
from os.path import exists
from PaddockTS.query import Query
from .utils import compute_ndvi, compute_ndwi, labels_to_paddocks, evaluate_paddocks
from .wnet import segment_wnet


def get_paddocks(
    query: Query,
    n_classes: int = 5,
    min_area_ha: float = 5,
    max_area_ha: float = 1500,
    min_compactness: float = 0.0,
    epsilon_factor: float = 0.005,
    method: str = 'rasterio',
    max_epochs: int = 1000,
    ncut_weight: float = 1.0,
):
    from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2

    if not exists(query.sentinel2_path):
        download_sentinel2(query)

    ds = xr.open_zarr(query.sentinel2_path, chunks=None)
    ndvi = compute_ndvi(ds)
    ndwi = compute_ndwi(ds)

    labels = segment_wnet(ndvi, ndwi, n_classes=n_classes, max_epochs=max_epochs, ncut_weight=ncut_weight)
    paddocks = labels_to_paddocks(labels, transform=ds.rio.transform(), crs=ds.rio.crs,
                                  min_area_ha=min_area_ha, max_area_ha=max_area_ha,
                                  min_compactness=min_compactness, epsilon_factor=epsilon_factor, method=method)

    metrics = evaluate_paddocks(paddocks, ndvi, ds.rio.transform())
    print(f'{metrics["n_paddocks"]} paddocks | '
          f'within var: {metrics["mean_within_variance"]:.4f} | '
          f'between var: {metrics["between_variance"]:.4f} | '
          f'ratio: {metrics["variance_ratio"]:.2f}')

    paddocks = paddocks.sort_values('area_ha', ascending=False).reset_index(drop=True)
    paddocks['label'] = range(1, len(paddocks) + 1)

    gpkg_path = f'{query.tmp_dir}/{query.stub}_paddocks_wnet.gpkg'
    paddocks.to_file(gpkg_path, driver='GPKG')
    print(f'Saved to {gpkg_path}')
    return paddocks, labels


def test():
    from PaddockTS.utils import get_example_query
    import matplotlib.pyplot as plt

    query = get_example_query()
    paddocks, labels = get_paddocks(query, n_classes=7)

    ds = xr.open_zarr(query.sentinel2_path, chunks=None)
    ndvi_median = np.nanmedian(compute_ndvi(ds), axis=2)
    x, y = ds.x.values, ds.y.values
    extent = [x.min(), x.max(), y.min(), y.max()]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].imshow(labels, cmap='tab20', extent=extent, origin='upper', interpolation='nearest')
    axes[0].set_title(f'W-Net Labels ({len(np.unique(labels))} classes)')
    axes[0].axis('off')
    axes[1].imshow(ndvi_median, cmap='RdYlGn', extent=extent, origin='upper')
    axes[1].set_title('Median NDVI')
    axes[1].axis('off')
    axes[2].imshow(ndvi_median, cmap='RdYlGn', extent=extent, origin='upper')
    paddocks.boundary.plot(ax=axes[2], color='red', linewidth=2)
    axes[2].set_title(f'{len(paddocks)} Paddocks (W-Net)')
    axes[2].axis('off')

    png_path = f'{query.tmp_dir}/{query.stub}_paddocks_wnet.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f'Saved plot to {png_path}')

    # per-paddock plots
    import geopandas as gpd
    n = len(paddocks)
    fig2, axes2 = plt.subplots(n, 3, figsize=(24, 8 * n))
    if n == 1:
        axes2 = axes2[np.newaxis, :]
    for row, (_, p) in enumerate(paddocks.iterrows()):
        minx, miny, maxx, maxy = p.geometry.bounds
        pad = max(maxx - minx, maxy - miny) * 0.2
        xlim, ylim = (minx - pad, maxx + pad), (maxy + pad, miny - pad)
        for col, (data, cmap) in enumerate([(labels, 'tab20'), (ndvi_median, 'RdYlGn'), (ndvi_median, 'RdYlGn')]):
            axes2[row, col].imshow(data, cmap=cmap, extent=extent, origin='upper', interpolation='nearest' if col == 0 else 'antialiased')
            axes2[row, col].set_xlim(xlim)
            axes2[row, col].set_ylim(ylim)
            axes2[row, col].axis('off')
        gpd.GeoSeries([p.geometry], crs=paddocks.crs).boundary.plot(ax=axes2[row, 2], color='red', linewidth=2)
        axes2[row, 0].set_ylabel(f'{p.area_ha:.0f} ha', rotation=0, labelpad=40)
        if row == 0:
            axes2[row, 0].set_title('Labels')
            axes2[row, 1].set_title('NDVI')
            axes2[row, 2].set_title('Paddock')

    plt.tight_layout()
    per_path = f'{query.tmp_dir}/{query.stub}_per_paddock_wnet.png'
    plt.savefig(per_path, dpi=150, bbox_inches='tight')
    print(f'Saved per-paddock plot to {per_path}')
    plt.show()


if __name__ == '__main__':
    test()
