import numpy as np
import rasterio
from pysheds.grid import Grid


def pysheds_accumulation(terrain_tif):
    """Read in the grid and dem and calculate the water flow direction and accumulation."""
    grid = Grid.from_raster(terrain_tif, nodata=np.float64(np.nan))
    dem = grid.read_raster(terrain_tif, nodata=np.float64(np.nan))

    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    fdir = grid.flowdir(inflated_dem, nodata_out=np.int64(0))
    acc = grid.accumulation(fdir, nodata_out=np.int64(0))

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
