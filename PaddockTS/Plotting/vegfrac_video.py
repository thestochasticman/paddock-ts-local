import cv2
import numpy as np
import xarray as xr
from PaddockTS.query import Query


def _to_rgb(ds, time_idx):
    """Map vegfrac to RGB: R=bg, G=pv, B=npv."""
    pv = ds['pv'].isel(time=time_idx).values.astype(np.float32)
    npv = ds['npv'].isel(time=time_idx).values.astype(np.float32)
    bg = ds['bg'].isel(time=time_idx).values.astype(np.float32)

    total = np.maximum(pv + npv + bg, 1e-6)
    pv, npv, bg = pv / total, npv / total, bg / total

    rgb = np.stack([bg, pv, npv], axis=-1)
    return np.nan_to_num(np.clip(rgb, 0, 1), nan=0.0)


def vegfrac_video(query: Query, fps: int = 4, min_size: int = 1080):
    ds = xr.open_zarr(query.vegfrac_path, chunks=None)
    n_times = ds.sizes['time']
    dates = ds.time.values
    h, w = ds.sizes['y'], ds.sizes['x']

    scale = max(1, min_size / max(h, w))
    out_h, out_w = int(h * scale) // 2 * 2, int(w * scale) // 2 * 2

    import os
    import subprocess
    import tempfile

    os.makedirs(query.out_dir, exist_ok=True)
    out_path = f'{query.out_dir}/{query.stub}_vegfrac.mp4'

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(n_times):
            rgb = _to_rgb(ds, i)
            frame = (rgb * 255).astype(np.uint8)
            if scale != 1:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

            date_str = str(np.datetime_as_string(dates[i], unit='D'))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = out_h / 600
            thickness = max(1, int(out_h / 400))
            (tw, th), _ = cv2.getTextSize(date_str, font, font_scale, thickness)
            x_pos = out_w - tw - 15
            y_pos = out_h - 15
            cv2.putText(frame, date_str, (x_pos, y_pos), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{tmpdir}/frame_{i:04d}.png', bgr)

        result = subprocess.run([
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', f'{tmpdir}/frame_%04d.png',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
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
    if not exists(query.vegfrac_path):
        from PaddockTS.IndicesAndVegFrac.veg_frac import compute_fractional_cover
        compute_fractional_cover(query)
    vegfrac_video(query)

if __name__ == '__main__':
    test()
