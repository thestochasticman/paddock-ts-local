import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PaddockTS.query import Query


def _to_rgb(ds, time_idx):
    """Extract an RGB image for a single timestep, scaled to 0-1."""
    r = ds['nbart_red'].isel(time=time_idx).values.astype(np.float32)
    g = ds['nbart_green'].isel(time=time_idx).values.astype(np.float32)
    b = ds['nbart_blue'].isel(time=time_idx).values.astype(np.float32)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[rgb == 0] = np.nan
    rgb /= 10000.0
    rgb = np.clip(rgb * 3, 0, 1)  # brighten
    rgb = np.nan_to_num(rgb, nan=0.0)
    return rgb


def sentinel2_video(query: Query, fps: int = 4, dpi: int = 150):
    ds = xr.open_zarr(query.sentinel2_path)
    n_times = ds.sizes['time']
    dates = ds.time.values
    h, w = ds.sizes['y'], ds.sizes['x']

    aspect = w / h
    fig = plt.figure(frameon=False)
    fig.set_size_inches(10 * aspect, 10)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    rgb0 = _to_rgb(ds, 0)
    im = ax.imshow(rgb0, origin='upper', aspect='auto')
    label = ax.text(0.98, 0.02, str(np.datetime_as_string(dates[0], unit='D')),
                    transform=ax.transAxes, color='white', fontsize=15,
                    ha='right', va='bottom', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, pad=3, edgecolor='none'))

    def update(i):
        im.set_data(_to_rgb(ds, i))
        label.set_text(str(np.datetime_as_string(dates[i], unit='D')))
        return [im, label]

    anim = animation.FuncAnimation(fig, update, frames=n_times, interval=1000 // fps, blit=True)

    out_path = f'{query.tmp_dir}/{query.stub}_sentinel2.mp4'
    anim.save(out_path, writer='ffmpeg', fps=fps, dpi=dpi)
    print(f'Saved video to {out_path}')
    plt.close(fig)
    return out_path


def test():
    from PaddockTS.utils import get_example_query
    sentinel2_video(get_example_query())

if __name__ == '__main__':
    test()
