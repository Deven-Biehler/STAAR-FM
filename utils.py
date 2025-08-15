import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.plot
from scipy.ndimage import generic_filter

def plot_window_on_dem(dem, window, title="DEM Window"):
    """Plot a specific window of the DEM."""
    

    fig, ax = plt.subplots(figsize=(10, 10))
    rasterio.plot.show(dem, ax=ax, title=title)
    
    # Draw the window rectangle
    rect = plt.Rectangle((window.col_off, window.row_off), window.width, window.height,
                         linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
    plt.show()


def downsample_dem(dem, scale_factor):
    """Downsample the DEM by a given scale factor."""
    if scale_factor <= 1:
        return dem
    
    # Downsample the DEM using slicing
    downsampled_dem = dem[::scale_factor, ::scale_factor]
    return downsampled_dem

def save_dem(data, path, crs):
    # Fill invalid values
    data = np.where(np.isnan(data), -9999, data)
    
    profile = {
        'driver': 'GTiff',
        'height': data.shape[0],
        'width': data.shape[1],
        'count': 1,
        'dtype': data.dtype,
        'crs': crs,
        'nodata': -9999  # Add this line
    }
    
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data, 1)


def roughness_index(dem):
    """Calculate roughness index using 3x3 neighborhood standard deviation."""
    
    def local_std(window):
        return np.std(window)
    
    # Apply 3x3 standard deviation filter
    roughness = generic_filter(dem, local_std, size=3, mode='reflect')
    return roughness