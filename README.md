# How to Use STAAR-FM

## Running the Demo
Run `demo.py` to visualize STAAR-FM running at multiple threshold values.

## Initialize the Adaptive DEM
```python
dynamic_dem = DynamicResolutionDEM(dem_path, scales, window)
```

## Generate Example Roughness Heuristic
```python
downscaled_dem = dynamic_dem_low.data[::scales[-1], ::scales[-1]]
roughness_low = roughness_index(downscaled_dem)
```

## Initialize STAAR-FM Flow Direction Calculation
```python
staar = STAAR(dynamic_dem)
```

## Calculate Flow Accumulation
```python
staar.calculate_flow_accumulation()
```

## Extract Flow Network
```python
staar.extract_flow_network()
```

## Access Results
- **Flow direction matrix**: `staar.fdir`
- **Flow accumulation matrix**: `staar.facc`
- **Flow network**: `staar.flow_network`
