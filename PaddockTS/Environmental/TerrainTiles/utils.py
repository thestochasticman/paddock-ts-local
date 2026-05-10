import numpy as np

# pysheds 0.5 calls np.in1d which was removed in NumPy 2.0. Alias to np.isin
# (the documented replacement, same semantics) before pysheds is imported.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

import rasterio
from pysheds.grid import Grid


def pysheds_accumulation(terrain_tif):
    """Read in the grid and dem and calculate the water flow direction and accumulation."""
    # Read raster to get nodata value from file
    with rasterio.open(terrain_tif) as src:
        nodata = src.nodata

    # Use file's nodata or a default for integer DEMs
    if nodata is None:
        nodata = -9999

    grid = Grid.from_raster(terrain_tif, nodata=nodata)
    dem = grid.read_raster(terrain_tif, nodata=nodata)

    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    fdir = grid.flowdir(inflated_dem, nodata_out=0)
    acc = grid.accumulation(fdir, nodata_out=0)

    return grid, dem, fdir, acc


def calculate_slope(terrain_tif):
    """Calculate the slope of a DEM in degrees."""
    with rasterio.open(terrain_tif) as src:
        dem = src.read(1)
        transform = src.transform
    gradient_y, gradient_x = np.gradient(dem, transform[4], transform[0])
    slope = np.arctan(np.sqrt(gradient_x**2 + gradient_y**2)) * (180 / np.pi)
    return slope


def calculate_twi(acc, slope):
    """Calculate topographic wetness index: TWI = ln(accumulation / tan(slope))."""
    ratio = acc / np.tan(np.radians(slope))
    ratio[ratio <= 0] = 1
    return np.log(ratio)


def calculate_aspect(terrain_tif):
    """Calculate aspect (direction of steepest descent) in degrees.

    Returns values 0-360 where:
    - 0/360 = North
    - 90 = East
    - 180 = South
    - 270 = West
    """
    with rasterio.open(terrain_tif) as src:
        dem = src.read(1)
        transform = src.transform

    gradient_y, gradient_x = np.gradient(dem, transform[4], transform[0])

    # Aspect in radians, then convert to degrees
    aspect_rad = np.arctan2(-gradient_x, gradient_y)
    aspect_deg = np.degrees(aspect_rad)

    # Convert from (-180, 180) to (0, 360)
    aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)

    return aspect_deg


def calculate_hli(slope, aspect, latitude):
    """Calculate Heat Load Index (McCune & Keon 2002).

    HLI accounts for the effect of slope and aspect on solar radiation.
    Higher values = more solar radiation = typically drier soils.

    Parameters
    ----------
    slope : ndarray
        Slope in degrees.
    aspect : ndarray
        Aspect in degrees (0=N, 90=E, 180=S, 270=W).
    latitude : float
        Latitude in degrees (negative for southern hemisphere).

    Returns
    -------
    ndarray
        Heat Load Index, range approximately 0-1.
    """
    # Convert to radians
    slope_rad = np.radians(slope)
    aspect_rad = np.radians(aspect)
    lat_rad = np.radians(latitude)

    # McCune & Keon (2002) equation
    # Folded aspect: transform so SW (225°) has highest value
    # In southern hemisphere, N-facing slopes get most radiation
    folded_aspect = np.abs(180 - np.abs(aspect - 225))
    folded_rad = np.radians(folded_aspect)

    # Simplified HLI formula
    hli = (
        np.exp(
            -1.467
            + 1.582 * np.cos(lat_rad) * np.cos(slope_rad)
            - 1.5 * np.cos(folded_rad) * np.sin(slope_rad) * np.sin(lat_rad)
            - 0.262 * np.sin(lat_rad) * np.sin(slope_rad)
            + 0.607 * np.sin(folded_rad) * np.sin(slope_rad)
        )
    )

    # Normalize to 0-1 range
    hli = np.clip(hli, 0, 1)

    return hli


def test():
    """Test terrain calculations and plot results."""
    import matplotlib.pyplot as plt
    from os.path import exists

    from PaddockTS.utils import get_example_query
    from PaddockTS.Environmental.TerrainTiles.download_terrain_tiles import (
        download_terrain,
        get_filename,
    )

    q = get_example_query()
    terrain_tif = get_filename(q)

    # Download if needed
    if not exists(terrain_tif):
        print('Downloading terrain...')
        download_terrain(q)

    print(f'Terrain file: {terrain_tif}')

    # Calculate all derivatives
    print('Calculating slope...')
    slope = calculate_slope(terrain_tif)

    print('Calculating aspect...')
    aspect = calculate_aspect(terrain_tif)

    print('Calculating flow accumulation (this may take a moment)...')
    grid, dem, fdir, acc = pysheds_accumulation(terrain_tif)

    print('Calculating TWI...')
    twi = calculate_twi(acc, slope)

    # Get latitude from bbox center for HLI
    lat_center = (q.bbox[1] + q.bbox[3]) / 2
    print(f'Calculating HLI (latitude={lat_center:.2f})...')
    hli = calculate_hli(slope, aspect, lat_center)

    # Print stats
    print(f'\nSlope: {np.nanmin(slope):.1f} - {np.nanmax(slope):.1f} degrees')
    print(f'Aspect: {np.nanmin(aspect):.1f} - {np.nanmax(aspect):.1f} degrees')
    print(f'TWI: {np.nanmin(twi):.1f} - {np.nanmax(twi):.1f}')
    print(f'HLI: {np.nanmin(hli):.3f} - {np.nanmax(hli):.3f}')

    # Plot 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # DEM
    im0 = axes[0, 0].imshow(dem, cmap='terrain')
    axes[0, 0].set_title('DEM (m)')
    plt.colorbar(im0, ax=axes[0, 0])

    # Slope
    im1 = axes[0, 1].imshow(slope, cmap='YlOrRd', vmin=0, vmax=45)
    axes[0, 1].set_title('Slope (degrees)')
    plt.colorbar(im1, ax=axes[0, 1])

    # Aspect
    im2 = axes[0, 2].imshow(aspect, cmap='hsv', vmin=0, vmax=360)
    axes[0, 2].set_title('Aspect (degrees)')
    plt.colorbar(im2, ax=axes[0, 2])

    # Flow accumulation (log scale)
    acc_log = np.log1p(acc)
    im3 = axes[1, 0].imshow(acc_log, cmap='Blues')
    axes[1, 0].set_title('Flow Accumulation (log)')
    plt.colorbar(im3, ax=axes[1, 0])

    # TWI
    twi_clipped = np.clip(twi, 0, 20)  # Clip extreme values for visualization
    im4 = axes[1, 1].imshow(twi_clipped, cmap='Blues', vmin=0, vmax=15)
    axes[1, 1].set_title('TWI (clipped 0-20)')
    plt.colorbar(im4, ax=axes[1, 1])

    # HLI
    im5 = axes[1, 2].imshow(hli, cmap='YlOrRd', vmin=0, vmax=1)
    axes[1, 2].set_title('Heat Load Index')
    plt.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()
    out_path = f'{q.tmp_dir}/terrain_test.png'
    plt.savefig(out_path, dpi=150)
    print(f'\nPlot saved: {out_path}')


if __name__ == '__main__':
    test()
