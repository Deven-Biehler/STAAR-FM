import math
import numpy as np
import rasterio
import time
from pysheds.grid import Grid


class DynamicResolutionDEM:
    def __init__(self, dem, scales, window):
        self.dem = dem
        self.scales = scales
        self.window = window
        
        self.resolution_map = None
        self.data = None
        self.crs = None
        self.transform = None
        self.load_dem()
    
    def load_dem(self):
        self.all_data = []
        with rasterio.open(self.dem) as src:
            geographic_window = rasterio.windows.bounds(self.window, src.transform)
            self.crs = src.crs
            self.transform = src.transform

        # Pre-processing
        grid = Grid.from_raster(self.dem, window=geographic_window)
        dem = grid.read_raster(self.dem, window=geographic_window)
        pit_filled_dem = grid.fill_pits(dem)
        flooded_dem = grid.fill_depressions(pit_filled_dem)
        inflated_dem = grid.resolve_flats(flooded_dem)
        self.data = inflated_dem

    def plot(self, title="Dynamic Resolution DEM", save=False):
        """
        Plot the DEM data with resolution map overlay.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        bounds = (self.window.col_off, self.window.col_off + self.window.width,
                 self.window.row_off, self.window.row_off + self.window.height)
        plt.imshow(self.data, cmap='terrain', extent=bounds, origin='upper')
        plt.colorbar(label='Elevation (m)')
        plt.title(title)
        plt.show()
        if save:
            plt.savefig(title.replace(" ", "_") + ".png")

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
            





class STAAR:
    def __init__(self, dynamic_DEM):
        self.dynamic_DEM = dynamic_DEM

        self.nodata = np.int32(-1)
        self.fdir_dtype = np.int32

        timers = {}
        timers['calculate_flow_direction'] = time.time()
        self.calculate_flow_direction()
        timers['calculate_flow_direction'] -= time.time()

    def get_elevation(self, data, kernel):
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
        start_time = time.time()
        kernel = np.array([[[-1,-1], [-1,0], [-1,1]],
                        [[0, -1], [0,0], [0, 1]],
                        [[1,-1], [1,0], [1,1]]])
        # Add padding to the data
        pad_width = max(self.dynamic_DEM.scales) // 2 + 1  # Padding width for boundary handling
        if np.issubdtype(self.dynamic_DEM.data.dtype, np.integer):
            max_val = np.iinfo(self.dynamic_DEM.data.dtype).max
        else:
            max_val = np.finfo(self.dynamic_DEM.data.dtype).max
        padded_data = np.pad(self.dynamic_DEM.data, pad_width=pad_width, mode='constant', constant_values=max_val)
        self.flow_dir = np.full((self.dynamic_DEM.data.shape[0], self.dynamic_DEM.data.shape[1]), -1, dtype=np.int32)
        # add padding to flow_dir
        self.flow_dir = np.pad(self.flow_dir, pad_width=pad_width, mode='constant', constant_values=-1)
        d8_codes = np.array([[32, 64, 128],
                            [16,  -1,   1],
                            [ 8,  4,   2]])
        center_idx = 4
        d8_distances = np.array([[math.sqrt(2), 1, math.sqrt(2)],
                                [1, -1, 1],
                                [math.sqrt(2), 1, math.sqrt(2)]])
        

        scales_arr = np.array(self.dynamic_DEM.scales) # (e.g. [1, 3, 9])
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

        for row in range(self.dynamic_DEM.resolution_map.shape[0]):
            for col in range(self.dynamic_DEM.resolution_map.shape[1]):
                row_prime = row * self.dynamic_DEM.scales[-1] + pad_width + self.dynamic_DEM.scales[-1] // 2  # Adjust for padding
                col_prime = col * self.dynamic_DEM.scales[-1] + pad_width + self.dynamic_DEM.scales[-1] // 2  # Adjust for padding

                scale = self.dynamic_DEM.resolution_map[row, col] # Get the scale for this window

                kernel_offsets = np.add(all_kernel_offsets[scale], np.array([[row_prime, col_prime]])) # Shape: (M*M, 3, 3, 2) where M is the composite scale
                elevations = self.get_elevation(padded_data, kernel_offsets) # Shape: (M*M, 3, 3)
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
        flow_dir_temp = "flow_dir_temp.tif"

        with rasterio.open(flow_dir_temp, 'w', driver='GTiff',
                        height=self.flow_dir.shape[0],
                        width=self.flow_dir.shape[1],
                        count=1, dtype=self.fdir_dtype,  # <-- use int32 for D8 codes
                        crs=self.dynamic_DEM.crs,
                        transform=self.dynamic_DEM.transform,
                        nodata=self.nodata) as dst:     # <-- set nodata to an unused int value
            dst.write(self.flow_dir.astype(self.fdir_dtype), 1)

        grid = Grid.from_raster(flow_dir_temp)
        fdir = grid.read_raster(data=flow_dir_temp, nodata=self.nodata, dtype=self.fdir_dtype)
        return grid, fdir

    def calculate_flow_accumulation(self):
        grid, fdir = self.get_pysheds_grid()
        self.facc = grid.accumulation(fdir, nodata_in=self.nodata, nodata_out=self.nodata)
    
    def extract_flow_network(self):
        grid, fdir = self.get_pysheds_grid()
        self.facc = grid.accumulation(fdir, nodata_in=self.nodata, nodata_out=self.nodata)
        # Convert top 5% of flow accumulation to river network
        mask = (self.facc > np.percentile(self.facc, 95))  # Boolean mask
        self.flow_network = grid.extract_river_network(fdir=fdir, mask=mask)