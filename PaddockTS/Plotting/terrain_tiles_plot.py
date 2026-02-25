from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter
from PaddockTS.Environmental.TerrainTiles.utils import pysheds_accumulation, calculate_slope, calculate_twi
from PaddockTS.query import Query
from PaddockTS.Environmental.TerrainTiles.download_terrain_tiles import get_filename
from os import makedirs
import tempfile
import os


def _array_to_tif(array, ref_path, out_path, dtype=None):
    """Write a numpy array to a GeoTIFF using another tif as the spatial reference."""
    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()
    if dtype:
        profile['dtype'] = dtype
    else:
        profile['dtype'] = array.dtype
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(array, 1)


def terrain_tiles_plot(query: Query, ds_sentinel2=None, sigma: int = 10):
    makedirs(query.out_dir, exist_ok=True)

    terrain_path = get_filename(query)

    with rasterio.open(terrain_path) as src:
        dem_raw = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        height, width = src.height, src.width

    # Smooth DEM for flow analysis
    dem_smooth = gaussian_filter(dem_raw.astype(float), sigma=sigma)
    tmpdir = tempfile.mkdtemp()
    smooth_path = os.path.join(tmpdir, 'smooth.tif')
    with rasterio.open(smooth_path, 'w', driver='GTiff', height=height, width=width,
                       count=1, dtype=dem_smooth.dtype, crs=crs, transform=transform,
                       nodata=nodata) as dst:
        dst.write(dem_smooth, 1)

    # Derive topographic variables in native CRS
    grid, dem, fdir, acc = pysheds_accumulation(smooth_path)
    slope = calculate_slope(smooth_path)
    twi = calculate_twi(acc, slope)

    # Write each derived variable as a georeferenced tif
    derived = {
        'accumulation': (acc, Resampling.max),
        'aspect': (fdir.astype('float64'), Resampling.nearest),
        'slope': (slope, Resampling.average),
        'twi': (twi, Resampling.average),
    }
    tif_paths = {}
    for name, (arr, _) in derived.items():
        path = os.path.join(tmpdir, f'{name}.tif')
        _array_to_tif(arr, smooth_path, path)
        tif_paths[name] = path

    # Load sentinel2 as the reference grid
    if ds_sentinel2 is None:
        if not os.path.exists(query.sentinel2_path):
            from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
            download_sentinel2(query)
        ds_ref = xr.open_zarr(query.sentinel2_path, chunks=None).isel(time=0)
    else:
        ds_ref = ds_sentinel2.isel(time=0) if 'time' in ds_sentinel2.dims else ds_sentinel2

    # Load terrain and derived tifs, reproject each to match sentinel2
    ds_terrain = rxr.open_rasterio(smooth_path).isel(band=0).drop_vars('band')
    ds = ds_terrain.rio.reproject_match(ds_ref, resampling=Resampling.average).to_dataset(name='terrain')

    for name, (_, resamp) in derived.items():
        da = rxr.open_rasterio(tif_paths[name]).isel(band=0).drop_vars('band')
        ds[name] = da.rio.reproject_match(ds_ref, resampling=resamp)

    # Clip 1 cell border (edge artifacts from flow algorithms)
    ds = ds.isel(y=slice(1, -1), x=slice(1, -1))

    # Reproject to EPSG:4326 for plotting
    ds_4326 = ds.rio.reproject('EPSG:4326')
    left, bottom, right, top = ds_4326.rio.bounds()
    extent = (left, right, bottom, top)

    dem_plot = ds_4326['terrain'].values
    acc_plot = ds_4326['accumulation'].values
    slope_plot = ds_4326['slope'].values

    # Map ArcGIS flow directions to sequential for aspect
    arcgis_dirs = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    sequential_dirs = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    fdir_raw = ds_4326['aspect'].values
    fdir_seq = np.zeros_like(fdir_raw)
    for a, s in zip(arcgis_dirs, sequential_dirs):
        fdir_seq[fdir_raw == a] = s

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax1, ax2), (ax3, ax4) = axes

    # Elevation
    im = ax1.imshow(dem_plot, cmap='terrain', interpolation='bilinear', extent=extent, origin='upper')
    ax1.set_title('Elevation')
    fig.colorbar(im, ax=ax1, label='m')
    levels = np.arange(np.floor(np.nanmin(dem_plot)), np.ceil(np.nanmax(dem_plot)), 10)
    if len(levels) > 1:
        contours = ax1.contour(dem_plot, levels=levels, colors='black',
                               linewidths=0.5, alpha=0.5, extent=extent, origin='upper')
        ax1.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')

    # Accumulation
    acc_safe = np.where(acc_plot > 0, acc_plot, 1)
    im = ax2.imshow(acc_safe, cmap='cubehelix', norm=colors.LogNorm(1, max(np.nanmax(acc_plot), 2)),
                    interpolation='bilinear', extent=extent, origin='upper')
    ax2.set_title('Flow Accumulation')
    fig.colorbar(im, ax=ax2, label='upstream cells')

    # Aspect
    im = ax3.imshow(fdir_seq, cmap='hsv', extent=extent, origin='upper')
    ax3.set_title('Aspect')
    cbar = fig.colorbar(im, ax=ax3)
    cbar.set_ticks(sequential_dirs)
    cbar.set_ticklabels(['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE'])

    # Slope
    im = ax4.imshow(slope_plot, cmap='Purples', extent=extent, origin='upper')
    ax4.set_title('Slope')
    fig.colorbar(im, ax=ax4, label='degrees')

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    plt.tight_layout()
    out_path = f'{query.out_dir}/{query.stub}_topography.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'  saved: {out_path}')
    return out_path


def test():
    from PaddockTS.utils import get_example_query
    from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
    from PaddockTS.Environmental.TerrainTiles.download_terrain_tiles import download_terrain
    q = get_example_query()
    # download_sentinel2(q)
    # download_terrain(q)
    terrain_tiles_plot(q)


if __name__ == '__main__':
    test()
