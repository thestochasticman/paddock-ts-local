from matplotlib import pyplot as plt
from rioxarray import open_rasterio
from os.path import basename
from os.path import dirname


def plot(
    bbox: list[float],
    variable: str,
    depth: str,
    filename: str
):
    ds = open_rasterio(filename)
    dirpath = dirname(filename)
    filename_base = basename(filename).strip('.tif')
    plot_name = f'{dirpath}/{filename_base}.png'
    fig, ax = plt.subplots(figsize=(8, 6))
    ds.isel(band=0).plot(ax=ax, cmap='YlOrRd')
    ax.set_title(f'{variable} content ({depth})')
    plt.tight_layout()
    plt.savefig(plot_name, dpi=150)
    