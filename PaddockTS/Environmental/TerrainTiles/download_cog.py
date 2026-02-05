import rasterio
from rasterio.windows import from_bounds, intersection
from rasterio.merge import merge
from rasterio.io import MemoryFile


def download_cog(bbox, url, filename):
    with rasterio.open(url) as src:
        full_window = from_bounds(*bbox, transform=src.transform)
        tile_window = from_bounds(*src.bounds, transform=src.transform)
        window = intersection(full_window, tile_window)

        data = src.read(1, window=window)
        transform = src.window_transform(window)
        meta = src.meta.copy()

    meta.update(
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        transform=transform,
    )

    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(data, 1)


def download_cogs(bbox, urls, filename):
    """Download and merge multiple COGs into a single clipped tif

    Parameters
    ----------
        bbox: [lon_min, lat_min, lon_max, lat_max] in EPSG:4326
        urls: List of HTTPS URLs to COGs
        filename: Output path for the merged tif
    """
    if len(urls) == 1:
        return download_cog(bbox, urls[0], filename)

    memfiles = []
    datasets = []
    for url in urls:
        with rasterio.open(url) as src:
            full_window = from_bounds(*bbox, transform=src.transform)
            tile_window = from_bounds(*src.bounds, transform=src.transform)
            window = intersection(full_window, tile_window)

            data = src.read(1, window=window)
            transform = src.window_transform(window)
            meta = src.meta.copy()
            meta.update(
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                transform=transform,
            )

        memfile = MemoryFile()
        with memfile.open(**meta) as dst:
            dst.write(data, 1)
        datasets.append(memfile.open())
        memfiles.append(memfile)

    merged, transform = merge(datasets, bounds=bbox)

    meta = datasets[0].meta.copy()
    meta.update(
        height=merged.shape[1],
        width=merged.shape[2],
        transform=transform,
    )

    for ds in datasets:
        ds.close()
    for mf in memfiles:
        mf.close()

    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(merged)