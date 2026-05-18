"""Fractional-cover video with paddock boundaries and labels overlaid.

Like :func:`PaddockTS.Plotting.fractional_cover_video.fractional_cover_video`,
but each frame has paddock polygon boundaries rasterised in red and the
paddock ID drawn at the polygon's representative point. The Sentinel-2
geotransform is used to align the paddocks to the cover grid (the two
share the same CRS and resolution).
"""

import cv2
import numpy as np
import xarray as xr
import rioxarray
from rasterio.features import rasterize
from PaddockTS.query import Query
from .fractional_cover_video import _to_rgb


def fractional_cover_paddocks_video(query: Query, paddocks_filepath: str | None = None, ds_fractional_cover=None, ds_sentinel2=None, fps: int = 4, min_size: int = 1080, label_col: str | None = None):
    """Encode a fractional-cover video with paddock outlines + labels.

    Args:
        query: The :class:`PaddockTS.query.Query`. Output is written to
            ``{query.out_dir}/{paddocks_stem}_fractional_cover_paddocks.mp4``.
        paddocks_filepath: Path to the paddocks file. If ``None``, uses
            SAM paddocks from ``{query.tmp_dir}/{query.stub}_sam_paddocks.gpkg``.
        ds_fractional_cover: Optional in-memory fractional cover dataset.
            If ``None``, opens (or generates, then opens)
            ``query.fractional_cover_path``.
        ds_sentinel2: Optional in-memory Sentinel-2 dataset, used only
            to read the rasterisation transform. If ``None``, opens
            ``query.sentinel2_path``.
        fps: Frames per second. Default 4.
        min_size: Minimum dimension of the output video in pixels.

    Returns:
        str: Filesystem path of the generated MP4.

    Raises:
        RuntimeError: If the ``ffmpeg`` invocation returns a non-zero
            exit code.
    """
    import os
    from pathlib import Path
    from PaddockTS.utils import load_user_paddocks

    # Default to SAM paddocks if no filepath provided
    if paddocks_filepath is None:
        paddocks_filepath = query.sam_paddocks_path

    out_stem = Path(paddocks_filepath).stem
    paddocks = load_user_paddocks(paddocks_filepath)

    if ds_fractional_cover is None:
        if not os.path.exists(query.fractional_cover_path):
            from PaddockTS.FractionalCover.compute_fractional_cover import compute_fractional_cover
            compute_fractional_cover(query)
        ds = xr.open_zarr(query.fractional_cover_path, chunks=None, decode_coords="all")
    else:
        ds = ds_fractional_cover
    n_times = ds.sizes['time']
    dates = ds.time.values
    h, w = ds.sizes['y'], ds.sizes['x']

    scale = max(1, min_size / max(h, w))
    out_h, out_w = int(h * scale) // 2 * 2, int(w * scale) // 2 * 2

    # rasterize boundaries once
    if ds_sentinel2 is None:
        from PaddockTS.Sentinel2.check_if_valid_clean_zarr_exists import check_if_valid_clean_zarr_exists
        if not check_if_valid_clean_zarr_exists(query.sentinel2_clean_path):
            from PaddockTS.Sentinel2.clean_sentinel2 import clean_sentinel2
            clean_sentinel2(query)
        s2 = xr.open_zarr(query.sentinel2_clean_path, chunks=None, decode_coords="all")
    else:
        s2 = ds_sentinel2

    # Reproject paddocks to match the dataset CRS
    ds_crs = s2.rio.crs
    if paddocks.crs != ds_crs:
        paddocks = paddocks.to_crs(ds_crs)

    transform = s2.rio.transform()
    shapes = [(geom.boundary, 1) for geom in paddocks.geometry]
    boundary_mask = rasterize(shapes, out_shape=(h, w), transform=transform, fill=0,
                              dtype=np.uint8, all_touched=True)
    if scale != 1:
        boundary_mask = cv2.resize(boundary_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    # precompute label positions
    inv_transform = ~transform
    label_positions = []
    for _, row in paddocks.iterrows():
        cx, cy = row.geometry.representative_point().x, row.geometry.representative_point().y
        px, py = inv_transform * (cx, cy)
        label = str(row[label_col]) if label_col else str(int(row.paddock))
        label_positions.append((label, int(px * scale), int(py * scale)))

    import os
    import subprocess
    import tempfile

    os.makedirs(query.out_dir, exist_ok=True)
    out_path = f'{query.out_dir}/{out_stem}_fractional_cover_paddocks.mp4'

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(n_times):
            rgb = _to_rgb(ds, i)
            frame = (rgb * 255).astype(np.uint8)
            if scale != 1:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            frame[boundary_mask > 0] = [255, 0, 0]

            font = cv2.FONT_HERSHEY_SIMPLEX

            # draw paddock labels
            label_scale = out_h / 1200
            label_thickness = max(1, int(out_h / 400))
            for label, lx, ly in label_positions:
                txt = str(label)
                (tw, th), _ = cv2.getTextSize(txt, font, label_scale, label_thickness)
                cv2.putText(frame, txt, (lx - tw // 2, ly + th // 2),
                            font, label_scale, (255, 0, 0), label_thickness, cv2.LINE_AA)

            date_str = str(np.datetime_as_string(dates[i], unit='D'))
            font_scale = out_h / 600
            thickness = max(1, int(out_h / 400))
            (tw, th), _ = cv2.getTextSize(date_str, font, font_scale, thickness)
            cv2.putText(frame, date_str, (out_w - tw - 15, th + 15),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{tmpdir}/frame_{i:04d}.png', bgr)

        result = subprocess.run([
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', f'{tmpdir}/frame_%04d.png',
            '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p',
            out_path,
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f'ffmpeg failed with code {result.returncode}')

    print(f'Saved video to {out_path}')
    return out_path


def test():
    from os.path import exists
    from PaddockTS.utils import get_example_query
    query = get_example_query()
    if not exists(query.fractional_cover_path):
        from PaddockTS.FractionalCover.compute_fractional_cover import compute_fractional_cover
        compute_fractional_cover(query)
    fractional_cover_paddocks_video(query)

if __name__ == '__main__':
    test()
