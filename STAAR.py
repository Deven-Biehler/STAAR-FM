import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LogNorm
from pysheds.grid import Grid
from shapely import buffer

class STAAR_Grid:
    """Class to handle dynamic resolution DEM and associated data."""

    def __init__(self, dem_path, scales, window, affine, crs):
        self.dem_path = dem_path
        self.scales = scales
        self.window = window
        self.affine = affine
        self.crs = crs

        self.resolution_map = None
        self.dem = None
        self.dem_grid = None
        self.bounds = None
        self.load_dem()
    
    def load_dem(self):
        self.all_data = []

        self.dem_grid = Grid.from_raster(self.dem_path, window=self.window, affine=self.affine)
        dem_raster = self.dem_grid.read_raster(self.dem_path, window=self.window)
        self.bounds = self.dem_grid.bbox

        # Pre-processing
        pit_filled_dem = self.dem_grid.fill_pits(dem_raster)
        flooded_dem = self.dem_grid.fill_depressions(pit_filled_dem)
        inflated_dem = self.dem_grid.resolve_flats(flooded_dem)
        self.dem = inflated_dem

    def calculate_resolution_map(self, heuristic_map, thresholds):
        """
        Calculate the resolution map based on the scales and the DEM data.
        The resolution map indicates which scale to use for each pixel in the DEM.
        """
        self.resolution_map = np.zeros((
            self.dem.shape[0] // self.scales[-1], 
            self.dem.shape[1] // self.scales[-1]), 
            dtype=np.int32
            )
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
        self.facc_raster = None
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
        if np.issubdtype(self.staar_grid.dem.dtype, np.integer):
            max_val = np.iinfo(self.staar_grid.dem.dtype).max
        else:
            max_val = np.finfo(self.staar_grid.dem.dtype).max
        padded_data = np.pad(self.staar_grid.dem, pad_width=pad_width, mode='constant', constant_values=max_val)
        self.fdir = np.full((self.staar_grid.dem.shape[0], self.staar_grid.dem.shape[1]), -1, dtype=np.int32)
        # add padding to flow_dir
        self.fdir = np.pad(self.fdir, pad_width=pad_width, mode='constant', constant_values=-1)
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

                self.fdir[all_coords[:, :, 0].flatten(), all_coords[:, :, 1].flatten()] = d8_encoded
        print("STAAR flow_dir time: ", time.time() - start_time)
    
    def get_fdir_grid(self):
        """Convert the flow direction to a PySheds Grid object by writing to a temporary file."""
        if self.fdir is None:
            raise ValueError("Flow direction must be calculated before creating the flow direction grid.")
        flow_dir_temp = "flow_dir_temp.tif"

        with rasterio.open(flow_dir_temp, 'w', driver='GTiff',
                        height=self.fdir.shape[0],
                        width=self.fdir.shape[1],
                        count=1, dtype=self.fdir_dtype,  # <-- use int32 for D8 codes
                        crs=self.staar_grid.crs,
                        transform=self.staar_grid.affine,
                        nodata=self.nodata) as dst:     # <-- set nodata to an unused int value
            dst.write(self.fdir.astype(self.fdir_dtype), 1)

        fdir_grid = Grid.from_raster(flow_dir_temp)
        fdir_raster = fdir_grid.read_raster(data=flow_dir_temp, nodata=self.nodata, dtype=self.fdir_dtype)
        return fdir_grid, fdir_raster

    def calculate_flow_accumulation(self):
        fdir_grid, fdir_raster = self.get_fdir_grid()
        self.facc_raster = fdir_grid.accumulation(fdir_raster, nodata_in=self.nodata, nodata_out=self.nodata)

    def extract_flow_network(self):
        if self.facc_raster is None:
            raise ValueError("Flow accumulation must be calculated before extracting the flow network.")
        # Convert top 5% of flow accumulation to river network
        mask = (self.facc_raster > np.percentile(self.facc_raster, 95))  # Boolean mask
        fdir_grid, fdir_raster = self.get_fdir_grid()
        self.flow_network = fdir_grid.extract_river_network(fdir=fdir_raster, mask=mask)

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import os
from matplotlib.colors import LogNorm
from matplotlib_scalebar.scalebar import ScaleBar
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


class STAAR_Plotter:
    def __init__(self, staar_fm):
        self.staar_fm = staar_fm
        
    def plot_flow_accumulation(self, use_log_scale=True, clip_percentile=99,
                              save=False, filename='flow_accumulation.png',
                              projection=ccrs.PlateCarree(),
                              title='Flow Accumulation'):
        """Plot flow accumulation with cartographic features."""
        if self.staar_fm.facc_raster is None:
            raise ValueError("Flow accumulation has not been calculated yet.")
        
        plot_data = self.staar_fm.facc_raster.copy()
        if use_log_scale:
            plot_data[plot_data <= 0] = 1
            norm = LogNorm(vmin=1, vmax=np.percentile(plot_data, clip_percentile))
        else:
            norm = None
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection=projection)

        bounds = self.staar_fm.staar_grid.bounds
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        ax.set_extent(extent, crs=projection)

        im = ax.imshow(plot_data, extent=extent, transform=projection,
                      cmap='Blues', norm=norm, origin='upper')
        
        gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        
        center_lat = (extent[2] + extent[3]) / 2
        meters_per_degree = 111320 * np.cos(np.radians(center_lat))
        scalebar = ScaleBar(meters_per_degree, location='lower left',
                           length_fraction=0.25, frameon=True)
        ax.add_artist(scalebar)
        
        ax.set_title(title + (' (log scale)' if use_log_scale else ''), 
                    fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, 
                    shrink=0.8, label='Flow Accumulation')
        
        if save:
            os.makedirs("plots", exist_ok=True)
            plt.savefig(f"plots/{filename}", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_flow_network(self, save=None, color='blue',
                         title="Extracted Flow Network",
                         projection=ccrs.PlateCarree()):
        """Plot flow network with cartographic features."""
        if self.staar_fm.flow_network is None:
            raise ValueError("Flow network has not been extracted yet.")
        
        fig = plt.figure(figsize=(12, 10))
        
        all_coords = []
        for feature in self.staar_fm.flow_network['features']:
            coords = feature['geometry']['coordinates']
            if feature['geometry']['type'] == 'LineString':
                all_coords.extend(coords)
            elif feature['geometry']['type'] == 'MultiLineString':
                for line in coords:
                    all_coords.extend(line)
        
        xs, ys = zip(*all_coords)
        bounds = self.staar_fm.staar_grid.bounds
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        
        ax = fig.add_subplot(111, projection=projection)
        ax.set_extent(extent, crs=projection)
        
        for feature in self.staar_fm.flow_network['features']:
            coords = feature['geometry']['coordinates']
            if feature['geometry']['type'] == 'LineString':
                xs, ys = zip(*coords)
                ax.plot(xs, ys, color=color, linewidth=1, transform=projection)
            elif feature['geometry']['type'] == 'MultiLineString':
                for line in coords:
                    xs, ys = zip(*line)
                    ax.plot(xs, ys, color=color, linewidth=1, transform=projection)

        gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        
        center_lat = (extent[2] + extent[3]) / 2
        meters_per_degree = 111320 * np.cos(np.radians(center_lat))
        scalebar = ScaleBar(meters_per_degree, location='lower left',
                           length_fraction=0.25, frameon=True)
        ax.add_artist(scalebar)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if save is None:
            plt.show()
        else:
            plt.savefig(save, bbox_inches='tight', dpi=300)
    
    def plot_flow_direction(self, save=None, 
                           title="Flow Direction Map",
                           projection=ccrs.PlateCarree()):
        """Plot the flow direction map with cartographic features."""
        if self.staar_fm.fdir is None:
            raise ValueError("Flow direction data is not available.")
        
        fig = plt.figure(figsize=(12, 8))
        
        bounds = self.staar_fm.staar_grid.bounds
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        
        ax = fig.add_subplot(111, projection=projection)
        ax.set_extent(extent, crs=projection)

        im = ax.imshow(self.staar_fm.fdir, extent=extent, transform=projection,
                      cmap='jet', vmin=1, vmax=128, origin='upper')
        
        gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # Add north arrow
        # Plot the N arrow
        right = extent[1] - 0.05 * (extent[1] - extent[0])  # 5% from right edge
        sbcy = extent[3] - 0.05 * (extent[3] - extent[2])  # 5% from top edge
        # buffer for text - white outline
        buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
        ax.text(right, sbcy, u'\u25B2\nN', transform=projection,
            horizontalalignment='center', verticalalignment='bottom',
            path_effects=buffer, zorder=2)

        
        
        center_lat = (extent[2] + extent[3]) / 2
        meters_per_degree = 111320 * np.cos(np.radians(center_lat))
        scalebar = ScaleBar(meters_per_degree, location='lower right',
                           length_fraction=0.25, frameon=True)
        ax.add_artist(scalebar)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02,
                    shrink=0.8, label='Flow Direction (D8 Codes)')
        
        if save:
            os.makedirs("plots", exist_ok=True)
            plt.savefig(f"plots/{save}", dpi=300, bbox_inches='tight')
        
        plt.show()