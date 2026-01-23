"""
Calendar plot generation for individual paddocks.

Generates weekly resampled RGB thumbnails for each paddock, organized by year and month.
Output is a JSON file with metadata and image paths for the web frontend.
"""

import xarray as xr
import numpy as np
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from typing import Optional
import json
from PIL import Image
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping
import warnings

warnings.filterwarnings('ignore')


def clip_ds_to_paddock(ds: xr.Dataset, paddock_geom, crs: str) -> xr.Dataset:
    """
    Clip an xarray Dataset to a single paddock geometry.

    Parameters:
        ds: xarray Dataset with 'x', 'y', 'time' dimensions
        paddock_geom: shapely geometry of the paddock
        crs: CRS of the dataset

    Returns:
        Clipped xarray Dataset
    """
    # Get bounds of the paddock
    minx, miny, maxx, maxy = paddock_geom.bounds

    # Clip to bounding box first
    ds_clipped = ds.sel(
        x=slice(minx, maxx),
        y=slice(maxy, miny)  # y is typically inverted
    )

    return ds_clipped


def resample_weekly(ds: xr.Dataset) -> xr.Dataset:
    """
    Resample dataset to weekly intervals with linear interpolation.

    Parameters:
        ds: xarray Dataset with 'time' dimension

    Returns:
        Weekly resampled dataset
    """
    return ds.resample(time="1W").interpolate("linear").interpolate_na(
        dim='time',
        method='linear'
    )


def normalize_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """
    Normalize RGB bands to 0-255 uint8.

    Parameters:
        red, green, blue: numpy arrays of band values

    Returns:
        RGB array of shape (H, W, 3) as uint8
    """
    rgb = np.dstack((red, green, blue)).astype('float32')

    # Robust normalization using percentiles
    p2 = np.nanpercentile(rgb, 2)
    p98 = np.nanpercentile(rgb, 98)

    if p98 > p2:
        rgb = (rgb - p2) / (p98 - p2)
    else:
        rgb = rgb / (np.nanmax(rgb) + 1e-10)

    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)

    # Replace NaN with black
    rgb = np.nan_to_num(rgb, nan=0)

    return rgb


def is_image_black(rgb_array: np.ndarray, threshold: float = 5.0) -> bool:
    """
    Check if an RGB image is essentially all black.

    Parameters:
        rgb_array: RGB numpy array of shape (H, W, 3)
        threshold: Mean pixel value below which the image is considered black

    Returns:
        True if the image is all black (or nearly all black)
    """
    return np.mean(rgb_array) < threshold


def save_thumbnail(rgb_array: np.ndarray, output_path: Path, size: tuple = (64, 64)) -> bool:
    """
    Save RGB array as a thumbnail image.

    Parameters:
        rgb_array: RGB numpy array of shape (H, W, 3)
        output_path: Path to save the image
        size: Thumbnail size (width, height)

    Returns:
        True if saved successfully, False if image was skipped (all black)
    """
    # Skip all-black images
    if is_image_black(rgb_array):
        return False

    img = Image.fromarray(rgb_array)
    img.thumbnail(size, Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, 'PNG')
    return True


def generate_paddock_calendar(
    ds: xr.Dataset,
    pol: gpd.GeoDataFrame,
    out_dir: str,
    stub: str,
    bands: list = ['nbart_red', 'nbart_green', 'nbart_blue'],
    thumbnail_size: tuple = (80, 80)
) -> dict:
    """
    Generate calendar plot data for all paddocks.

    Creates weekly resampled RGB thumbnails for each paddock, organized by year/month.

    Parameters:
        ds: xarray Dataset with RGB bands and 'time' dimension
        pol: GeoDataFrame with paddock polygons (must have 'paddock' column)
        out_dir: Output directory for images and JSON
        stub: Job stub/identifier
        bands: List of band names [red, green, blue]
        thumbnail_size: Size of thumbnail images (width, height)

    Returns:
        Dictionary with calendar data structure
    """
    out_path = Path(out_dir)
    calendar_dir = out_path / 'calendar'
    calendar_dir.mkdir(parents=True, exist_ok=True)

    # Resample to weekly
    print("Resampling to weekly intervals...")
    ds_weekly = resample_weekly(ds)

    # Get CRS from dataset
    crs = ds.attrs.get('crs', 'EPSG:6933')

    # Ensure polygons are in correct CRS
    pol = pol.to_crs(crs)

    # Add paddock column if not present (same as get_paddock_ts.py)
    if 'paddock' not in pol.columns:
        pol['paddock'] = range(1, len(pol) + 1)

    # Structure: { paddock_id: { year: { month: [ {date, image_path} ] } } }
    calendar_data = {
        'meta': {
            'stub': stub,
            'start_date': str(ds_weekly.time.values[0])[:10],
            'end_date': str(ds_weekly.time.values[-1])[:10],
            'thumbnail_size': list(thumbnail_size),
            'total_weeks': len(ds_weekly.time),
        },
        'paddocks': {}
    }

    # Process each paddock
    for idx, row in pol.iterrows():
        paddock_id = str(row.get('paddock', idx))
        print(f"Processing paddock {paddock_id}...")

        paddock_geom = row.geometry

        # Clip dataset to paddock bounds
        ds_paddock = clip_ds_to_paddock(ds_weekly, paddock_geom, crs)

        if ds_paddock.sizes['x'] == 0 or ds_paddock.sizes['y'] == 0:
            print(f"  Skipping paddock {paddock_id} - no data in bounds")
            continue

        paddock_data = {}

        # Process each time step
        for t_idx, t in enumerate(ds_paddock.time.values):
            timestamp = str(t)[:10]  # YYYY-MM-DD
            dt_obj = datetime.fromisoformat(timestamp)
            year = str(dt_obj.year)
            month = str(dt_obj.month).zfill(2)

            # Initialize year/month structure
            if year not in paddock_data:
                paddock_data[year] = {}
            if month not in paddock_data[year]:
                paddock_data[year][month] = []

            # Extract RGB for this timestep
            try:
                red = ds_paddock[bands[0]].isel(time=t_idx).values
                green = ds_paddock[bands[1]].isel(time=t_idx).values
                blue = ds_paddock[bands[2]].isel(time=t_idx).values

                # Normalize and create RGB
                rgb = normalize_rgb(red, green, blue)

                # Save thumbnail (skip if all black)
                img_filename = f"{paddock_id}_{timestamp}.png"
                img_path = calendar_dir / paddock_id / year / month / img_filename
                saved = save_thumbnail(rgb, img_path, thumbnail_size)

                # Only add to calendar data if image was saved (not black)
                if saved:
                    rel_path = f"calendar/{paddock_id}/{year}/{month}/{img_filename}"
                    paddock_data[year][month].append({
                        'date': timestamp,
                        'week': dt_obj.isocalendar()[1],
                        'image': rel_path
                    })

            except Exception as e:
                print(f"  Error processing {timestamp}: {e}")
                continue

        # Clean up empty months and years before adding to calendar data
        cleaned_paddock_data = {}
        for year, months in paddock_data.items():
            cleaned_months = {m: imgs for m, imgs in months.items() if imgs}
            if cleaned_months:
                cleaned_paddock_data[year] = cleaned_months

        calendar_data['paddocks'][paddock_id] = cleaned_paddock_data

    # Save calendar data JSON
    json_path = out_path / f'{stub}_calendar.json'
    with open(json_path, 'w') as f:
        json.dump(calendar_data, f, indent=2)

    print(f"Calendar data saved to {json_path}")

    return calendar_data


def generate_calendar_from_query(query, device='cpu'):
    """
    Generate calendar plots from a Query object.

    This integrates with the existing PaddockTS pipeline.

    Parameters:
        query: PaddockTS Query object
        device: Device for processing ('cpu' or 'cuda')
    """
    import pickle

    # Load the preprocessed dataset
    with open(query.path_ds2, 'rb') as f:
        ds = pickle.load(f)

    # Load paddock polygons
    pol = gpd.read_file(query.path_polygons)

    # Generate calendar data
    return generate_paddock_calendar(
        ds=ds,
        pol=pol,
        out_dir=query.stub_out_dir,
        stub=query.stub,
        bands=['nbart_red', 'nbart_green', 'nbart_blue'],
        thumbnail_size=(80, 80)
    )


def test():
    """Test function for calendar plot generation."""
    from PaddockTS.query import get_example_query
    query = get_example_query()
    generate_calendar_from_query(query)


if __name__ == '__main__':
    test()
