import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LogNorm
from pysheds.grid import Grid

class STAAR_Grid:
    """Class to handle dynamic resolution DEM and associated data."""
    def __init__(self, dem, scales, window):
        self.dem = dem
        self.scales = scales
        self.window = window

        self.resolution_map = None
        self.data = None
        self.crs = None
        self.transform = None
        self.dem_grid = None
        self.load_dem()
    
    def load_dem(self):
        self.all_data = []
        with rasterio.open(self.dem) as src:
            geographic_window = rasterio.windows.bounds(self.window, src.transform)
            self.crs = src.crs
            self.transform = src.transform
            # Apply transform to window
            self.transform = rasterio.windows.transform(self.window, src.transform)
            self.data = src.read(1, window=self.window)

        # Pre-processing
        self.dem_grid = Grid.from_raster(self.dem, window=geographic_window, affine=self.transform)
        dem = self.dem_grid.read_raster(self.dem, window=geographic_window)
        pit_filled_dem = self.dem_grid.fill_pits(dem)
        flooded_dem = self.dem_grid.fill_depressions(pit_filled_dem)
        inflated_dem = self.dem_grid.resolve_flats(flooded_dem)
        self.data = inflated_dem

    def calculate_resolution_map(self, heuristic_map, thresholds):
        """
        Calculate the resolution map based on the scales and the DEM data.
        The resolution map indicates which scale to use for each pixel in the DEM.
        """
        self.resolution_map = np.zeros((self.data.shape[0] // self.scales[-1], self.data.shape[1] // self.scales[-1]), dtype=np.int32)
        if heuristic_map.shape != self.resolution_map.shape:
            raise ValueError("Heuristic map shape must match the resolution map shape.")
        
        thresholds = np.percentile(heuristic_map, [0] + list(thresholds) + [100])
        
        
        self.resolution_map.fill(self.scales[-1])  # Initialize with the largest scale
        for i, scale in enumerate(self.scales[::-1]):
            # Create a mask for the current scale based on the heuristic map
            mask = (heuristic_map >= thresholds[i]) & (heuristic_map < thresholds[i + 1]) # if heuristic_map is greater than or equal to the lower threshold and less than the upper threshold
            self.resolution_map[mask] = scale
        # Ensure the resolution map is of type int32
        self.resolution_map = self.resolution_map.astype(np.int32)
            


class STAAR_FlowModeling:
    """Custom implementation of D8 flow direction using STAAR_Grid."""

    def __init__(self, STAAR_grid):
        self.staar_grid = STAAR_grid

        self.nodata = np.int32(-1)
        self.fdir_dtype = np.int32
        
        self.fdir = None
        self.facc = None
        self.fdir_grid = None
        self.flow_network = None

    def _get_elevation(self, data, kernel):
        x_coords = kernel[:, :, :, 0]  # Shape: (3, 3, 9)
        y_coords = kernel[:, :, :, 1]  # Shape: (3, 3, 9)
        # Flatten, index, then reshape back
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()

        # Read all points
        values_flat = data[x_flat, y_flat]

        # Reshape back to desired shape
        result = values_flat.reshape(len(kernel), 3, 3)
        return result

    def calculate_flow_direction(self):
        """Vectorized D8 flow direction calculation using STAAR multi-resolution grid."""
        start_time = time.time()
        kernel = np.array([[[-1,-1], [-1,0], [-1,1]],
                        [[0, -1], [0,0], [0, 1]],
                        [[1,-1], [1,0], [1,1]]])
        # Add padding to the data
        pad_width = max(self.staar_grid.scales) // 2 + 1  # Padding width for boundary handling
        if np.issubdtype(self.staar_grid.data.dtype, np.integer):
            max_val = np.iinfo(self.staar_grid.data.dtype).max
        else:
            max_val = np.finfo(self.staar_grid.data.dtype).max
        padded_data = np.pad(self.staar_grid.data, pad_width=pad_width, mode='constant', constant_values=max_val)
        self.flow_dir = np.full((self.staar_grid.data.shape[0], self.staar_grid.data.shape[1]), -1, dtype=np.int32)
        # add padding to flow_dir
        self.flow_dir = np.pad(self.flow_dir, pad_width=pad_width, mode='constant', constant_values=-1)
        d8_codes = np.array([[32, 64, 128],
                            [16,  -1,   1],
                            [ 8,  4,   2]])
        center_idx = 4
        d8_distances = np.array([[math.sqrt(2), 1, math.sqrt(2)],
                                [1, -1, 1],
                                [math.sqrt(2), 1, math.sqrt(2)]])
        

        scales_arr = np.array(self.staar_grid.scales) # (e.g. [1, 3, 9])
        all_d8_distances = {scale: d8_distances * (scale//2+1) for scale in scales_arr} # D8 distances scaled by each resolution
        composite_scales = np.divide(scales_arr.max(),  scales_arr).astype(int) # counts how many times each sub voxel fits into the max voxel
        all_kernels = {scale: kernel*(scale//2+1) for scale in scales_arr} # Contains the kernel sizes
        all_kernel_offsets = {} # Contains the actual coordinates
        # Initialize kernel offsets for each scale
        for scale_id, composite_scale in enumerate(composite_scales):
            scale = scales_arr[scale_id]
            composite_points = [(i*scale, j*scale) for i in range(composite_scale) for j in range(composite_scale)]
            all_kernel_offsets[scale] = all_kernels[scale] + np.stack(composite_points)[:, np.newaxis, np.newaxis, :]
        meshgrid_cache = {}
        for scale in scales_arr:
            half_size = scale // 2
            offsets = np.arange(-half_size, half_size + 1)
            row_offsets, col_offsets = np.meshgrid(offsets, offsets, indexing='ij')
            meshgrid_cache[scale] = np.column_stack([row_offsets.ravel(), col_offsets.ravel()])

        for row in range(self.staar_grid.resolution_map.shape[0]):
            for col in range(self.staar_grid.resolution_map.shape[1]):
                row_prime = row * self.staar_grid.scales[-1] + pad_width + self.staar_grid.scales[-1] // 2  # Adjust for padding
                col_prime = col * self.staar_grid.scales[-1] + pad_width + self.staar_grid.scales[-1] // 2  # Adjust for padding

                scale = self.staar_grid.resolution_map[row, col] # Get the scale for this window

                kernel_offsets = np.add(all_kernel_offsets[scale], np.array([[row_prime, col_prime]])) # Shape: (M*M, 3, 3, 2) where M is the composite scale
                elevations = self._get_elevation(padded_data, kernel_offsets) # Shape: (M*M, 3, 3)
                center_points = elevations[:, 1, 1] # Shape: (M*M,)
                elevation_diffs = elevations - center_points[:, np.newaxis, np.newaxis] # Shape: (M*M, 3, 3)

                slopes = (-elevation_diffs / all_d8_distances[scale]) # Shape: (M*M, 3, 3)
                slopes_flat = slopes.reshape(len(slopes), -1)  # (M*M, 9)
                max_vals = np.max(slopes_flat, axis=1)  # (M*M,)
                center_tied = slopes_flat[:, 4] == max_vals  # (M*M,)
                flat_argmax = np.argmax(slopes_flat, axis=1)  # (M*M,)
                flat_argmax[center_tied] = center_idx  # Set center index for tied values
                row_indices, col_indices = np.unravel_index(flat_argmax, (3, 3)) # Shape: (M*M,)
                d8_encoded = d8_codes[row_indices, col_indices] # Shape: (M*M,)
                d8_encoded = np.repeat(d8_encoded, scale**2) # Shape: (M*M*scale^2,)
                center_coords = kernel_offsets[:, 1, 1, :] # Center of the kernel (M, 2)
                offsets_flat = meshgrid_cache[scale] # (M*M, 2)
                all_coords = center_coords[:, None, :] + offsets_flat[None, :, :] # Shape: (M, M*M, 2)

                self.flow_dir[all_coords[:, :, 0].flatten(), all_coords[:, :, 1].flatten()] = d8_encoded
        print("STAAR flow_dir time: ", time.time() - start_time)
    
    def get_pysheds_grid(self):
        """Convert the flow direction to a PySheds Grid object by writing to a temporary file."""
        flow_dir_temp = "flow_dir_temp.tif"

        with rasterio.open(flow_dir_temp, 'w', driver='GTiff',
                        height=self.flow_dir.shape[0],
                        width=self.flow_dir.shape[1],
                        count=1, dtype=self.fdir_dtype,  # <-- use int32 for D8 codes
                        crs=self.staar_grid.crs,
                        transform=self.staar_grid.transform,
                        nodata=self.nodata) as dst:     # <-- set nodata to an unused int value
            dst.write(self.flow_dir.astype(self.fdir_dtype), 1)

        self.fdir_grid = Grid.from_raster(flow_dir_temp)
        self.fdir = self.fdir_grid.read_raster(data=flow_dir_temp, nodata=self.nodata, dtype=self.fdir_dtype)
        return self.fdir_grid, self.fdir

    def calculate_flow_accumulation(self):
        self.fdir_grid, self.fdir = self.get_pysheds_grid()
        self.facc = self.fdir_grid.accumulation(self.fdir, nodata_in=self.nodata, nodata_out=self.nodata)

    def extract_flow_network(self):
        if self.facc is None:
            raise ValueError("Flow accumulation must be calculated before extracting the flow network.")
        # Convert top 5% of flow accumulation to river network
        mask = (self.facc > np.percentile(self.facc, 95))  # Boolean mask
        self.flow_network = self.fdir_grid.extract_river_network(fdir=self.fdir, mask=mask)

class STAAR_Plotter:
    def __init__(self, staar_fm):
        self.staar_fm = staar_fm

    def plot_flow_accumulation(self, use_log_scale=True, clip_percentile=99, 
                          save=False, filename='flow_accumulation.png'):
        """Plot flow accumulation using rasterio's built-in plotting."""
        from rasterio.plot import show
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        plot_data = self.staar_fm.facc.copy()
        if use_log_scale:
            plot_data[plot_data <= 0] = 1
            norm = LogNorm(vmin=1, vmax=np.percentile(plot_data, clip_percentile))
        else:
            norm = None
        
        # Use rasterio's show with the grid's transform
        im = show(plot_data, transform=self.staar_fm.fdir_grid.affine, ax=ax,
                cmap='Blues', norm=norm)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Flow Accumulation' + (' (log scale)' if use_log_scale else ''))
        
        # Add colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im.get_images()[0], cax=cax, label='Flow Accumulation')
        
        if save:
            os.makedirs("plots", exist_ok=True)
            plt.savefig(f"plots/{filename}", dpi=300, bbox_inches='tight')
        
        plt.show()


    def plot_flow_network(self, flow_network, output=None, color='blue'):
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

    def plot_flow_direction(self, flow_dir, save=None):
        """
        Plot the flow direction map.
        """
        from rasterio.plot import show
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    
        _, ax = plt.subplots(figsize=(12, 8))
        im = show(flow_dir, transform=self.staar_fm.staar_grid.transform, ax=ax,
                cmap='jet', vmin=1, vmax=128)
        plt.title("Flow Direction Map")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im.get_images()[0], cax=cax, label='Flow Direction (D8 Codes)')
    
        if save:
            os.makedirs("plots", exist_ok=True)
            plt.savefig(f"plots/{save}", dpi=300, bbox_inches='tight')
    
        plt.show()