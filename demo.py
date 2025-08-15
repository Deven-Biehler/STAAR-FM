import numpy as np
import rasterio
from time import time

from STAAR import STAAR, DynamicResolutionDEM
from dem_plotting import plot_flow_accumulation, plot_flow_direction, plot_flow_network
from pysheds.grid import Grid
from utils import roughness_index


# Save the original can_cast function
_original_can_cast = np.can_cast

def patched_can_cast(from_, to, casting='safe'):
    """
    Patched version of np.can_cast that handles Python scalars
    by converting them to numpy scalars first (NEP 50 compatibility)
    """
    # If from_ is a Python scalar, convert it to numpy scalar
    if isinstance(from_, (bool, int, float, complex)):
        from_ = np.array(from_).dtype
    elif hasattr(from_, 'dtype'):
        from_ = from_.dtype
    elif not isinstance(from_, (np.dtype, type, str)):
        try:
            from_ = np.array(from_).dtype
        except:
            pass
    
    return _original_can_cast(from_, to, casting=casting)

# Apply the monkey patch
np.can_cast = patched_can_cast



def preprocess_dem(dem_path, window):
    with rasterio.open(dem_path) as src:
        geographic_window = rasterio.windows.bounds(window, src.transform)
    grid = Grid.from_raster(dem_path, window=geographic_window)
    dem = grid.read_raster(dem_path, window=geographic_window)
    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)
    # Save the processed DEM
    print("Processed DEM saved.")
    return grid, inflated_dem


def run_pysheds_flow_accumulation(dem_path, window):
    grid, inflated_dem = preprocess_dem(dem_path, window)
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    start_time = time()
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap, nodata_out=np.int64(-1))
    print(f"PySheds Flow Direction calculated in {time() - start_time} seconds")
    acc = grid.accumulation(fdir, dirmap=dirmap, nodata_out=np.int64(-1))
    return fdir, acc


def main(threshold):
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Preprocessing 
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    dem_path = "data/sa_dem_3s.tif"
    scales = [1, 3]  # Example scales
    threshold_low = [threshold]
    threshold_high = [0]
    patch_size = 1
    with rasterio.open(dem_path) as src:
        height, width = src.height, src.width
        window = rasterio.windows.Window(height * (.5 - patch_size/200), width * (.5 - patch_size/200), height * patch_size/100, width * patch_size/100) # Grab a chunk out of middle
        # Crop window to multiple of 3
        window = window.round(3)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Method
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    dynamic_dem_low = DynamicResolutionDEM(dem_path, scales, window)
    downscaled_dem = dynamic_dem_low.data[::scales[-1], ::scales[-1]]
    roughness_low = roughness_index(downscaled_dem)
    dynamic_dem_low.calculate_resolution_map(roughness_low, threshold_low)
    staar_low = STAAR(dynamic_dem_low)
    staar_low.calculate_flow_accumulation()
    staar_low.extract_flow_network()


    dynamic_dem_high = DynamicResolutionDEM(dem_path, scales, window)
    downscaled_dem = dynamic_dem_high.data[::scales[-1], ::scales[-1]]
    roughness_high = roughness_index(downscaled_dem)
    dynamic_dem_high.calculate_resolution_map(roughness_high, threshold_high)
    staar_high = STAAR(dynamic_dem_high)
    staar_high.calculate_flow_accumulation()
    staar_high.extract_flow_network()

    fdir, facc = run_pysheds_flow_accumulation(dem_path, window)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OUTPUT 
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("number of pixels:", dynamic_dem_high.data.size)
    print("Threshold:", threshold)

    print("STAAR Flow (low-res):")
    plot_flow_direction(staar_low.flow_dir)
    plot_flow_accumulation(staar_low.facc)
    plot_flow_network(staar_low.flow_network, color='red')

    print("STAAR Flow (high-res):")
    plot_flow_direction(staar_high.flow_dir)
    plot_flow_accumulation(staar_high.facc)
    plot_flow_network(staar_high.flow_network, color='blue')

    print("PySheds Flow:")
    plot_flow_direction(fdir)
    plot_flow_accumulation(facc)


    unique_resolutions, counts = np.unique(dynamic_dem_low.resolution_map, return_counts=True)
    percents = counts / np.sum(counts) * 100
    
    print("Resolution distribution (% of total pixels):")
    for resolution, percent in zip(unique_resolutions, percents):
        print(f"  Resolution {resolution}: {percent:.2f}%")


    # Calculate RMSE between high and low:
    rmse = np.sqrt(np.mean((staar_high.facc - staar_low.facc) ** 2))
    print(f"RMSE between high and low resolution: {rmse:.2f}")

    print('\a')
        

if __name__ == "__main__":
    for percentile in range(0, 101, 10):
        main(percentile)