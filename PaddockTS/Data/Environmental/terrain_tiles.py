"""
Download elevation data from the MapZen Terrain Tiles API.

Terrain Tiles documentation: https://github.com/tilezen/joerd/blob/master/docs/data-sources.md
"""
import os
import subprocess

import numpy as np
import rasterio
import xarray as xr
import rioxarray as rxr  # noqa: F401
from scipy.interpolate import griddata
from pyproj import Transformer

from PaddockTS.query import Query


def _transform_bbox(bbox, inputEPSG="EPSG:4326", outputEPSG="EPSG:3857"):
    transformer = Transformer.from_crs(inputEPSG, outputEPSG)
    x1, y1 = transformer.transform(bbox[1], bbox[0])
    x2, y2 = transformer.transform(bbox[3], bbox[2])
    return (x1, y1, x2, y2)


def _generate_xml(tile_level=14, filename="terrain_tiles.xml"):
    xml_string = f"""<GDAL_WMS>
  <Service name="TMS">
    <ServerUrl>https://s3.amazonaws.com/elevation-tiles-prod/geotiff/${{z}}/${{x}}/${{y}}.tif</ServerUrl>
  </Service>
  <DataWindow>
    <UpperLeftX>-20037508.34</UpperLeftX>
    <UpperLeftY>20037508.34</UpperLeftY>
    <LowerRightX>20037508.34</LowerRightX>
    <LowerRightY>-20037508.34</LowerRightY>
    <TileLevel>{tile_level}</TileLevel>
    <TileCountX>1</TileCountX>
    <TileCountY>1</TileCountY>
    <YOrigin>top</YOrigin>
  </DataWindow>
  <Projection>EPSG:3857</Projection>
  <BlockSizeX>512</BlockSizeX>
  <BlockSizeY>512</BlockSizeY>
  <BandsCount>1</BandsCount>
  <DataType>Int16</DataType>
  <ZeroBlockHttpCodes>403,404</ZeroBlockHttpCodes>
  <DataValues>
    <NoData>-32768</NoData>
  </DataValues>
  <Cache/>
</GDAL_WMS>"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(xml_string)


def _run_gdalwarp(bbox, filename, tmpdir, tile_level=14, debug=False, verbose=True):
    """Use gdalwarp to download a tif from terrain tiles."""
    if os.path.exists(filename):
        os.remove(filename)

    xml_path = os.path.join(tmpdir, "terrain_tiles.xml")
    _generate_xml(tile_level, xml_path)

    bbox_3857 = _transform_bbox(bbox)
    min_x, min_y, max_x, max_y = bbox_3857
    command = [
        "gdalwarp",
        "-of", "GTiff",
        "-te", str(min_x), str(min_y), str(max_x), str(max_y),
        xml_path, filename
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if debug:
        print("Terrain Tiles STDOUT:", result.stdout, flush=True)
        print("Terrain Tiles STDERR:", result.stderr, flush=True)
    if verbose:
        print(f"Downloaded {filename}")


def _interpolate_nan(filename):
    """Fix bad measurements in terrain tiles dem."""
    with rasterio.open(filename) as dataset:
        dem = dataset.read(1)
        meta = dataset.meta.copy()

    threshold = 10
    heights = sorted(set(dem.flatten()))
    if len(heights) <= 1:
        return dem, meta

    lowest_correct_height = min(heights)
    for i in range(len(heights) // 2 - 1, -1, -1):
        if heights[i + 1] - heights[i] > threshold:
            lowest_correct_height = heights[i + 1]
            break
    Z = np.where(dem < lowest_correct_height, np.nan, dem)

    x_coords, y_coords = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    z_flat = Z.flatten()

    mask = ~np.isnan(z_flat)
    x_flat = x_flat[mask]
    y_flat = y_flat[mask]
    z_flat = z_flat[mask]
    xy_coords = np.vstack((x_flat, y_flat), dtype=float).T

    X, Y = np.meshgrid(
        np.linspace(0, Z.shape[1] - 1, Z.shape[1]),
        np.linspace(0, Z.shape[0] - 1, Z.shape[0])
    )
    nearest = griddata(xy_coords, z_flat, (X, Y), method='nearest')
    return nearest, meta


def _save_dem(dem, meta, filename, verbose=True):
    meta.update({
        "driver": "GTiff",
        "height": dem.shape[0],
        "width": dem.shape[1],
        "count": 1,
        "dtype": dem.dtype
    })
    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(dem, 1)
    if verbose:
        print(f"Saved {filename}")


def _create_xarray(dem, meta):
    """Convert the cleaned dem into an xarray."""
    transform = meta['transform']
    height = meta['height']
    width = meta['width']
    crs = meta['crs']

    x_coords = transform.c + np.arange(width) * transform.a
    y_coords = transform.f + np.arange(height) * transform.e

    dem_da = xr.DataArray(
        dem,
        dims=("y", "x"),
        coords={"x": x_coords, "y": y_coords},
        attrs={
            "crs": crs.to_string(),
            "transform": transform,
            "nodata": meta["nodata"]
        },
        name="terrain"
    )
    return dem_da.to_dataset()


def terrain_tiles(query: Query, tile_level=14, interpolate=True, verbose=True):
    """Download 10m resolution elevation from terrain_tiles.

    Parameters
    ----------
        query: Query object with lat, lon, buffer, stub_out_dir, stub, stub_tmp_dir
        tile_level: The zoom level to determine the pixel size in the resulting tif.
        interpolate: Boolean flag to decide whether to try to fix bad values or not.

    Returns
    -------
        xarray.Dataset with elevation data, or None if interpolate=False
    """
    from os import makedirs

    lat, lon, buffer = query.lat, query.lon, query.buffer
    outdir, stub, tmpdir = query.stub_out_dir, query.stub, query.stub_tmp_dir
    makedirs(outdir, exist_ok=True)
    makedirs(tmpdir, exist_ok=True)

    if verbose:
        print("Starting terrain_tiles")

    buffer = max(0.00002, buffer)

    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]
    filename = os.path.join(tmpdir, f"{stub}_terrain_original.tif")
    _run_gdalwarp(bbox, filename, tmpdir, tile_level, verbose=verbose)

    if interpolate:
        dem, meta = _interpolate_nan(filename)
        filename = os.path.join(outdir, f"{stub}_terrain.tif")
        _save_dem(dem, meta, filename, verbose=verbose)
        ds = _create_xarray(dem, meta)
    else:
        ds = None

    return ds
