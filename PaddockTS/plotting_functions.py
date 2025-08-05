import xarray as xr
from shapely.geometry import Polygon
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import matplotlib.pyplot as plt

def plot_indices_timeseries(ds, out_dir, stub):
    """
    Generate a grid of timeseries plots for selected variables from an xarray Dataset and save the figure.

    Parameters:
        ds (xarray.Dataset): Input dataset containing a 'time' dimension and various data variables.
        out_dir (str): Directory path where the output figure will be saved.
        stub (str): String to be appended to the output filename.

    The function selects all variable names that do not start with "nbart" and are not "bg", "pv", or "npv".
    The figure is saved as: out_dir + stub + '_indices_ts.tif'
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Generate list of variables to plot:
    indices = [var for var in ds.data_vars
               if not var.startswith("nbart") and var not in ['bg', 'pv', 'npv']]
    
    n_indices = len(indices)
    n_times = ds.time.size

    # Create a grid: rows = variables, columns = time steps.
    fig, axes = plt.subplots(n_indices, n_times, 
                             figsize=(3 * n_times, 3 * n_indices), 
                             squeeze=False)

    for i, idx in enumerate(indices):
        # Compute global min and max for the current variable over all time steps.
        data_all = ds[idx].values  # shape: (time, y, x)
        global_vmin = np.nanmin(data_all)
        global_vmax = np.nanmax(data_all)

        im_list = []
        for j, t in enumerate(ds.time.values):
            ax = axes[i, j]
            data = ds[idx].sel(time=t).values
            im = ax.imshow(data, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
            ax.axis('off')
            im_list.append(im)

            # Set title for the top row panels.
            if i == 0:
                ax.set_title(str(t)[:10], fontsize=10)

        # Add a colorbar to the right of the right-most panel in this row.
        ax_last = axes[i, -1]
        divider = make_axes_locatable(ax_last)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im_list[-1], cax=cax)

        # Add the variable name to the left of the left-most panel.
        ax_first = axes[i, 0]
        ax_first.text(-0.15, 0.5, idx, transform=ax_first.transAxes,
                      va='center', ha='right', fontsize=12, rotation=90)

    plt.tight_layout()

    # Save the figure.
    output_filename = f"{out_dir}{stub}_indices_ts.tif"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()

def plot_paddock_map_auto_fourier(raster_path, pol, out_dir, stub):
    """
    Reads a three-band Fourier Transform TIFF file, overlays paddock polygons with labels,
    and saves & displays the resulting map.

    Parameters:
        raster_path (str): Path to the input TIFF raster file.
        pol (geopandas.GeoDataFrame): GeoDataFrame containing paddock polygons and a 'paddock' column for labels.
        stub (str): A string to be included in the output filename.
        out_dir (str): Directory path where the output file will be saved.

    The figure is saved as: out_dir + stub + '_paddock_map_auto_fourier.tif'
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import rasterio

    # Read the raster and convert to an RGB image.
    with rasterio.open(raster_path) as src:
        # Read the three bands
        band1 = src.read(1)  # Red
        band2 = src.read(2)  # Green
        band3 = src.read(3)  # Blue

        # Stack the bands into an RGB image
        rgb = np.dstack((band1, band2, band3))
        rgb = rgb.astype('float32')
        rgb /= rgb.max()  # Normalize to 0-1

        # Save the raster bounds and CRS for later use.
        bounds = src.bounds
        raster_crs = src.crs

    # Ensure the polygon GeoDataFrame uses the same CRS as the raster.
    pol = pol.to_crs(raster_crs)

    # Plotting: create a figure and axes.
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the RGB image with the correct spatial extent.
    ax.imshow(rgb, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))

    # Overlay the paddock polygons.
    pol.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)

    # Add labels at the centroid of each paddock polygon.
    for idx, row in pol.iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, row['paddock'],
                fontsize=12, ha='center', va='center', color='yellow')

    # Save the figure to file.
    output_filename = f"{out_dir}{stub}_paddock_map_auto_fourier.tif"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')

    # Remove axis and show the figure.
    plt.axis('off')
    plt.show()

def plot_paddock_map_auto_rgb(ds, pol, out_dir, stub):
    """
    Create an RGB composite image from an xarray Dataset using the nbart_* bands,
    composite over the time axis (using the median), overlay labelled paddock polygons,
    and save & display the resulting map.

    Parameters:
        ds (xarray.Dataset): Dataset containing a 'time' dimension and nbart_red, nbart_green, and nbart_blue bands.
        pol (geopandas.GeoDataFrame): GeoDataFrame with paddock polygons and a 'paddock' column for labels.
        stub (str): String to be appended to the output filename.
        out_dir (str): Directory path where the output figure will be saved.

    The figure is saved as: out_dir + stub + '_paddock_map_auto_rgb.tif'
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Create a composite over time using the median for each nbart band.
    red = ds['nbart_red'].median(dim='time').values
    green = ds['nbart_green'].median(dim='time').values
    blue = ds['nbart_blue'].median(dim='time').values

    # Stack the bands to form an RGB image.
    rgb = np.dstack((red, green, blue)).astype('float32')
    rgb /= np.nanmax(rgb)  # Normalize to 0-1

    # Determine spatial extent from the dataset's x and y coordinates.
    left   = float(ds.x.min().values)
    right  = float(ds.x.max().values)
    bottom = float(ds.y.min().values)
    top    = float(ds.y.max().values)

    # Reproject the polygons to the dataset's CRS if available.
    if 'crs' in ds.attrs:
        pol = pol.to_crs(ds.attrs['crs'])

    # Create the plot.
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, extent=(left, right, bottom, top))

    # Overlay the paddock polygons.
    # pol.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)

    list_coords_pol = [polygon.exterior for polygon in pol.geometry]

    for coords in list_coords_pol:
        x, y = coords.xy
        ax.plot(x, y, color='red')

    # Add paddock labels at the centroid of each polygon.
    for _, row in pol.iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, row['paddock'],
                fontsize=12, ha='center', va='center', color='yellow')

    # Save the figure.
    output_filename = f"{out_dir}{stub}_paddock_map_auto_rgb.tif"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')

    # Hide axes and display the figure.
    plt.axis('off')
    plt.show()

def animate_paddock_map_auto_rgb(
    ds: xr.Dataset,
    pol: Polygon,
    bands: list[str],
    out_dir: str,
    stub: str,
    width_pixels: int = 600,
    fps: int = 10,
    dpi: int = 100
):
    # 1. Compute figure size & aspect ratio
    ny, nx = ds.sizes['y'], ds.sizes['x']
    aspect = ny / nx
    fig_w, fig_h = width_pixels / dpi, (width_pixels * aspect) / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.axis('off')

    left   = float(ds.x.min().values)
    right  = float(ds.x.max().values)
    bottom = float(ds.y.min().values)
    top    = float(ds.y.max().values)
    
    # 2. Compute the median composite reference and its max
    #    to serve as our fixed normalizer.
    red_med   = ds['nbart_red'  ].median(dim='time').values.astype(float)
    green_med = ds['nbart_green'].median(dim='time').values.astype(float)
    blue_med  = ds['nbart_blue' ].median(dim='time').values.astype(float)

    ref_rgb = np.dstack((red_med, green_med, blue_med))
    normalizer = np.nanmax(ref_rgb)
    if normalizer == 0 or np.isnan(normalizer):
        raise ValueError("Reference composite max is zero or NaN; cannot normalize.")

    print(f"Normalizing all frames by reference max = {normalizer:.3f}")

    # 3. Helper to build and normalize one frame
    def make_frame(idx: int) -> np.ndarray:
        layers = []
        for band in bands:
            arr = ds[band].isel(time=idx).values.astype(float)
            # Normalize by the reference composite max
            arr = arr / normalizer
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            arr = np.clip(arr, 0.0, 1.0)
            # to uint8
            layers.append((arr * 255).astype(np.uint8))
        return np.stack(layers, axis=-1)
    
   
    # 4. Initialize the first frame
    pol.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)
    im = ax.imshow(make_frame(0), origin='upper', aspect='equal', extent=(left, right, bottom, top))
    if 'crs' in ds.attrs:
        pol = pol.to_crs(ds.attrs['crs'])
    


    list_coords_pol = [tuple(polygon.exterior.coords) for polygon in pol.geometry]
    # print(list_coords_pol[0])

    def _update(frame_idx: int):
        im.set_data(make_frame(frame_idx))
        
        return (im,)

    
    anim = FuncAnimation(fig, _update, frames=ds.sizes['time'], blit=True)
    output_path = f"{out_dir}/{stub}_manpad_RGB.mp4"
    anim.save(output_path, writer=FFMpegWriter(fps=fps))
    plt.close(fig)
    

def plot_silo_daily(silo, ds, out_dir, stub):
    """
    Create a figure with three panels showing:
      1. Daily Rain Time Series as a bar plot with downward pointing arrows indicating "Sentinel-2 observation".
      2. Daily Temperature Range (legend shows only the min and max temperatures).
      3. Actual vs. Potential Evapotranspiration.
    
    In the top panel, a downward pointing arrow is plotted at each time point from ds.
    Each arrow spans 1/8 of the y-axis range at the top of the panel.
    
    The figure is saved as:
        out_dir + stub + '_silo_daily.tif'
    
    Parameters:
        silo (xarray.Dataset): SILO daily data containing variables 'daily_rain', 
                               'min_temp', 'max_temp', 'et_morton_actual', 
                               'et_morton_potential', and 'time'.
        ds (xarray.Dataset): Dataset whose time coordinate provides the dates to mark with arrows.
        out_dir (str): Directory path where the output figure will be saved.
        stub (str): String to be included in the output filename.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Extract the time and variable values from the SILO dataset.
    time = silo['time'].values
    daily_rain = silo['daily_rain'].values
    min_temp = silo['min_temp'].values
    max_temp = silo['max_temp'].values
    et_actual = silo['et_morton_actual'].values
    et_potential = silo['et_morton_potential'].values

    # Create a figure with three vertically stacked subplots sharing the same x-axis.
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    # Top panel: Daily Rain Time Series as a bar plot (without a label).
    ax1.bar(time, daily_rain, color='blue')
    ax1.set_ylabel('Daily Rain (mm)')
    ax1.set_title('Daily Rain Time Series')

    # Force drawing of the figure to ensure correct axis limits for arrow placement.
    fig.canvas.draw()
    ymin, ymax = ax1.get_ylim()
    segment_height = (ymax - ymin) / 8

    # Plot a downward pointing arrow at each time point from ds on the top panel.
    for t in ds['time'].values:
        ax1.annotate('',
                     xy=(t, ymax),
                     xytext=(t, ymax - segment_height),
                     arrowprops=dict(facecolor='grey', edgecolor='grey', arrowstyle='<|-', lw=1))
    
    # Create a proxy artist for the arrow to include in the legend.
    arrow_proxy = Line2D([0], [0], marker=r'$\downarrow$', color='grey', linestyle='None',
                           markersize=10, label='Sentinel-2 observation')
    # Set the legend with a modified position (upper right but slightly lower).
    ax1.legend(handles=[arrow_proxy], labels=['Sentinel-2 observation'], 
               loc='upper right', bbox_to_anchor=(1, 0.9))

    # Middle panel: Daily Temperature Range.
    ax2.fill_between(time, min_temp, max_temp, color='lightblue', alpha=0.5)
    ax2.plot(time, min_temp, color='blue', label='Min Temperature')
    ax2.plot(time, max_temp, color='red', label='Max Temperature')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('Daily Temperature Range')
    ax2.legend()

    # Bottom panel: Actual vs. Potential Evapotranspiration.
    ax3.plot(time, et_actual, color='green', label='Actual ET')
    ax3.plot(time, et_potential, color='orange', label='Potential ET')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('ET (mm/day)')
    ax3.set_title('Actual vs. Potential Evapotranspiration')
    ax3.legend()

    plt.tight_layout()
    output_filename = f"{out_dir}{stub}_silo_daily.tif"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

### This is basically replacing the above:
def plot_env_ts(silo, ds, Ssoil, out_dir, stub):
    """
    Create a figure with four panels showing:
      1. Daily Rain Time Series as a bar plot with downward pointing arrows 
         indicating Sentinel-2 observation dates.
      2. Soil Moisture (Ssoil) time series (averaged across the geographic region).
      3. Daily Temperature Range (with min and max temperatures).
      4. Actual vs. Potential Evapotranspiration.
    
    The x-axis for all panels is based on the daily time axis from the SILO dataset.
    
    The figure is saved as:
        out_dir + stub + '_env_ts.tif'
    
    Parameters:
        silo (xarray.Dataset): SILO daily data containing variables 'daily_rain', 'min_temp', 
                               'max_temp', 'et_morton_actual', 'et_morton_potential', and 'time'.
        ds (xarray.Dataset): Dataset whose time coordinate provides the dates for Sentinel-2 observations.
        Ssoil (xarray.DataArray): Soil moisture time series data (averaged over the geographic region)
                                  with an 8-day frequency.
        out_dir (str): Directory path where the output figure will be saved.
        stub (str): String to be included in the output filename.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Extract the time axis and data from the SILO dataset.
    time_daily = silo['time'].values
    daily_rain = silo['daily_rain'].values
    min_temp = silo['min_temp'].values
    max_temp = silo['max_temp'].values
    et_actual = silo['et_morton_actual'].values
    et_potential = silo['et_morton_potential'].values

    # Create a figure with four vertically stacked subplots sharing the same x-axis.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    # Panel 1: Daily Rain Time Series.
    ax1.bar(time_daily, daily_rain, color='blue')
    ax1.set_ylabel('Daily Rain (mm)')
    ax1.set_title('Daily Rain Time Series')

    # Draw the figure to get the correct axis limits for arrow placement.
    fig.canvas.draw()
    ymin, ymax = ax1.get_ylim()
    segment_height = (ymax - ymin) / 8

    # Plot a downward pointing arrow at each Sentinel-2 observation time from ds.
    for t in ds['time'].values:
        ax1.annotate('',
                     xy=(t, ymax),
                     xytext=(t, ymax - segment_height),
                     arrowprops=dict(facecolor='grey', edgecolor='grey', arrowstyle='<|-', lw=1))
    
    # Create a proxy artist for the arrow to include in the legend.
    arrow_proxy = Line2D([0], [0], marker=r'$\downarrow$', color='grey', linestyle='None',
                           markersize=10, label='Sentinel-2 observation')
    ax1.legend(handles=[arrow_proxy], loc='upper right', bbox_to_anchor=(1, 0.9))

    # Panel 2: Soil Moisture (Ssoil) Time Series.
    # Although Ssoil has an 8-day frequency, the x-axis is set to match the daily SILO time.
    ax2.plot(Ssoil['time'].values, Ssoil.values, color='purple', label='Soil Moisture')
    ax2.set_ylabel('Soil Moisture (units)')
    ax2.set_title('Soil Moisture Time Series')
    ax2.legend()

    # Panel 3: Daily Temperature Range.
    ax3.fill_between(time_daily, min_temp, max_temp, color='lightblue', alpha=0.5)
    ax3.plot(time_daily, min_temp, color='blue', label='Min Temperature')
    ax3.plot(time_daily, max_temp, color='red', label='Max Temperature')
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_title('Daily Temperature Range')
    ax3.legend()

    # Panel 4: Actual vs. Potential Evapotranspiration.
    ax4.plot(time_daily, et_actual, color='green', label='Actual ET')
    ax4.plot(time_daily, et_potential, color='orange', label='Potential ET')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('ET (mm/day)')
    ax4.set_title('Actual vs. Potential Evapotranspiration')
    ax4.legend()

    # Ensure all panels share the same x-axis range based on the SILO daily time.
    ax1.set_xlim(time_daily[0], time_daily[-1])

    plt.tight_layout()
    output_filename = f"{out_dir}{stub}_env_ts.tif"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
