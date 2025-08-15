import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LogNorm

def plot_dem(dem, title='Digital Elevation Model', cmap='terrain', save=False, filename='dem_plot.png'):
    plt.figure(figsize=(12, 8))
    dem = np.clip(dem, np.percentile(dem, 1), np.percentile(dem, 98))
    plt.imshow(dem, cmap=cmap)
    plt.colorbar(label='Elevation (m)')
    plt.title(title)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    if save:
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{filename}", dpi=300, bbox_inches='tight')
    
    plt.show()



def plot_flow_accumulation(flow_data, title='Flow Accumulation', cmap='Blues',
                           save=False, filename='flow_accumulation_plot.png',
                           use_log_scale=True, clip_percentile=99):
    """Plot flow accumulation data."""
    
    plt.figure(figsize=(12, 8))

    # Prepare data for visualization
    plot_data = flow_data.copy()
    if use_log_scale:
        # Ensure positive values for log scale
        plot_data[plot_data <= 0] = 1
        norm = LogNorm(vmin=1, vmax=np.percentile(plot_data, clip_percentile))
        im = plt.imshow(plot_data, cmap=cmap, norm=norm)
    else:
        vmax = np.percentile(plot_data, clip_percentile)
        im = plt.imshow(plot_data, cmap=cmap, vmin=0, vmax=vmax)

    # Add colorbar and labels
    plt.colorbar(im, label='Flow Accumulation' + (' (log scale)' if use_log_scale else ''))
    plt.title(title)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')

    # Save or show the plot
    if save:
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{filename}", dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_flow_network(flow_network, output=None, color='blue'):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    for feature in flow_network['features']:
        coords = feature['geometry']['coordinates']
        if feature['geometry']['type'] == 'LineString':
            xs, ys = zip(*coords)
            plt.plot(xs, ys, color=color, linewidth=1)
        elif feature['geometry']['type'] == 'MultiLineString':
            for line in coords:
                xs, ys = zip(*line)
                plt.plot(xs, ys, color='blue', linewidth=1)
    plt.title("Extracted Flow Network")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis('equal')
    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches='tight')

def plot_flow_direction(flow_dir, output=None):
    """
    Plot the flow direction map.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(flow_dir, cmap='jet', vmin=0, vmax=255)
    plt.colorbar(label='Flow Direction (D8 Codes)')
    plt.title("Flow Direction Map")
    plt.axis('off')
    if output is not None:
        plt.savefig(output, bbox_inches='tight')
    else:
        plt.show()
    plt.close()