# How to use
Running demo.py to visualize STAAR-FM running at multiple threshold values.

Initialize the adaptive DEM with:
dynamic_dem = DynamicResolutionDEM(dem_path, scales, window)

Generate an example roughness heuristic:
downscaled_dem = dynamic_dem_low.data[::scales[-1], ::scales[-1]]
roughness_low = roughness_index(downscaled_dem)

Initialize STAAR-FM flow direction calculation:
staar = STAAR(dynamic_dem)
To calculate flow accumulation:
staar.calculate_flow_accumulation()
To extract flow network:
staar.extract_flow_network()

flow direction matrix: staar.fdir
flow accumulation matrix: staar.facc
flow network: staar.flow_network
