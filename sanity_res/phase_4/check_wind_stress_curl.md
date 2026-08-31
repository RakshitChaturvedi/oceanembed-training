
==========================================================================================
PHASE 4 — WIND STRESS CURL PRE-FLIGHT DIAGNOSTIC
==========================================================================================
File: /home/rakshitchaturvedi/Desktop/Projects/oceanembed-training/data/processed/phase2/wind.nc
File exists: PASS

------------------------------------------------------------------------------------------
1. DATASET
------------------------------------------------------------------------------------------
<xarray.Dataset> Size: 414MB
Dimensions:     (time: 1419, latitude: 101, longitude: 241)
Coordinates:
  * time        (time) datetime64[ns] 11kB 2021-01-01 2021-01-02 ... 2024-11-19
  * latitude    (latitude) float64 808B 5.0 5.25 5.5 5.75 ... 29.5 29.75 30.0
  * longitude   (longitude) float64 2kB 45.0 45.25 45.5 ... 104.5 104.8 105.0
Data variables:
    wind_u      (time, latitude, longitude) float32 138MB ...
    wind_v      (time, latitude, longitude) float32 138MB ...
    wind_speed  (time, latitude, longitude) float32 138MB ...
Attributes: (12/63)
    contact:                         Remote Sensing Systems, support@remss.com
    Conventions:                     CF-1.7 ACDD-1.3
    data_structure:                  grid
    title:                           RSS CCMP V3.1 6-hourly surface winds (Le...
    summary:                         RSS VAM 6-hour analyses using ERA-5 wind...
    institute_id:                    RSS
    ...                              ...
    oceanembed_regrid_method:        bilinear
    oceanembed_grid_resolution:      0.25 degrees
    oceanembed_phase:                phase2
    oceanembed_temporal_resolution:  daily
    oceanembed_spatial_domain:       5.0N-30.0N, 45.0E-105.0E
    oceanembed_spatial_resolution:   0.25deg

Dimensions:
  time        : 1419
  latitude    : 101
  longitude   : 241

Variables:
  wind_u
  wind_v
  wind_speed

------------------------------------------------------------------------------------------
2. VARIABLE VALIDATION
------------------------------------------------------------------------------------------
Expected: ['wind_u', 'wind_v', 'wind_speed']
Actual:   ['wind_u', 'wind_v', 'wind_speed']
Variable set: PASS

wind_u
  dims:  ('time', 'latitude', 'longitude')
  shape: (1419, 101, 241)
  dtype: float32
  units: m s-1
  dimensions: PASS

wind_v
  dims:  ('time', 'latitude', 'longitude')
  shape: (1419, 101, 241)
  dtype: float32
  units: m s-1
  dimensions: PASS

wind_speed
  dims:  ('time', 'latitude', 'longitude')
  shape: (1419, 101, 241)
  dtype: float32
  units: m s-1
  dimensions: PASS

------------------------------------------------------------------------------------------
3. TIME
------------------------------------------------------------------------------------------
Start: 2021-01-01T00:00:00.000000000
End:   2024-11-19T00:00:00.000000000
Count: 1419
Duplicates: 0
Duplicates: PASS
Unique day intervals: [1]
Daily spacing: PASS

------------------------------------------------------------------------------------------
4. HORIZONTAL GRID
------------------------------------------------------------------------------------------
Latitude:
  First: 5.0
  Last:  30.0
  Count: 101
  Min spacing: 0.25
  Max spacing: 0.25
  Unique spacing: [0.25]

Longitude:
  First: 45.0
  Last:  105.0
  Count: 241
  Min spacing: 0.25
  Max spacing: 0.25
  Unique spacing: [0.25]

Grid validation:
  Latitude increasing: PASS
  Longitude increasing: PASS
  Latitude grid:  PASS
  Longitude grid: PASS

------------------------------------------------------------------------------------------
5. WIND DATA QUALITY
------------------------------------------------------------------------------------------

wind_u
  Total:   34,539,879
  NaNs:    0 (0.0000%)
  Infs:    0 (0.0000%)
  Zeros:   964,921 (2.7936%)
  Min:     -20.150171279907227
  Max:     26.10869598388672
  Mean:    0.7083775997161865
  Median:  0.1708410382270813
  Std:     3.496674060821533
  dtype:   float32

wind_v
  Total:   34,539,879
  NaNs:    0 (0.0000%)
  Infs:    0 (0.0000%)
  Zeros:   964,920 (2.7936%)
  Min:     -21.52005958557129
  Max:     26.70372772216797
  Mean:    0.2833138406276703
  Median:  0.04998159408569336
  Std:     3.2182798385620117
  dtype:   float32

wind_speed
  Total:   34,539,879
  NaNs:    0 (0.0000%)
  Infs:    0 (0.0000%)
  Zeros:   964,920 (2.7936%)
  Min:     0.0
  Max:     28.337739944458008
  Mean:    3.744935989379883
  Median:  2.979471445083618
  Std:     3.0235037803649902
  dtype:   float32

------------------------------------------------------------------------------------------
6. UNIT VALIDATION
------------------------------------------------------------------------------------------
wind_u: m s-1
wind_v: m s-1
wind_speed: m s-1

Wind units: PASS

------------------------------------------------------------------------------------------
7. U/V MASK CONSISTENCY
------------------------------------------------------------------------------------------
U NaNs: 0
V NaNs: 0
U/V NaN masks identical: PASS

------------------------------------------------------------------------------------------
8. SPATIAL MISSINGNESS
------------------------------------------------------------------------------------------
U spatial cells containing any NaN: 0 / 24,341 (0.00%)
V spatial cells containing any NaN: 0 / 24,341 (0.00%)

U permanently NaN cells: 0
V permanently NaN cells: 0
U intermittently NaN cells: 0
V intermittently NaN cells: 0

------------------------------------------------------------------------------------------
9. POSSIBLE LAND / MASK STRUCTURE
------------------------------------------------------------------------------------------
This section checks whether permanently missing cells form large coherent spatial regions.

Permanent missing cells: 0 / 24,341

Permanent missing cells by latitude:

Permanent missing cells by longitude:

------------------------------------------------------------------------------------------
10. ZERO-VALUE STRUCTURE
------------------------------------------------------------------------------------------
U zero values: 964,921 (2.7936%)
V zero values: 964,920 (2.7936%)
U=0 AND V=0: 964,920 (2.7936%)
Spatial cells containing at least one U=V=0: 680 / 24,341
Spatial cells permanently U=V=0: 680

------------------------------------------------------------------------------------------
11. PHYSICAL SANITY
------------------------------------------------------------------------------------------

wind_u
  Allowed range: [-100.0, 100.0]
  Below range:   0
  Above range:   0
  Result: PASS

wind_v
  Allowed range: [-100.0, 100.0]
  Below range:   0
  Above range:   0
  Result: PASS

wind_speed
  Allowed range: [0.0, 100.0]
  Below range:   0
  Above range:   0
  Result: PASS

------------------------------------------------------------------------------------------
12. WIND SPEED CONSISTENCY
------------------------------------------------------------------------------------------
Valid points: 34,539,879
Maximum absolute error: 1.8160863497485025e-06
Mean absolute error:    9.387173044744522e-08
Result: PASS

------------------------------------------------------------------------------------------
13. GEOGRAPHIC DERIVATIVE SPACING
------------------------------------------------------------------------------------------
Latitude resolution:  0.25 degrees
Longitude resolution: 0.25 degrees

Meridional spacing dy: 27,798.73 m

Zonal spacing dx:
  At 5.00°N:  27,692.95 m
  At 17.50°N: 26,512.12 m
  At 30.00°N: 24,074.41 m

IMPORTANT:
  dx varies with latitude.
  The curl implementation must NOT use the same x-distance for every latitude.

------------------------------------------------------------------------------------------
14. NUMERICAL DIFFERENTIATION TEST
------------------------------------------------------------------------------------------
Testing xarray.differentiate() on a known analytical field.

Latitude derivative interior max error: 0.000000e+00
Latitude derivative boundary max error: 0.000000e+00

Interpretation:
  Interior points use centered finite differences.
  Boundary points use one-sided finite differences.

------------------------------------------------------------------------------------------
15. DOMAIN EDGE DIAGNOSTIC
------------------------------------------------------------------------------------------
Domain boundaries:
  South: 5.0°N
  North: 30.0°N
  West:  45.0°E
  East:  105.0°E

wind_u boundary statistics:
  south : min=   0.000, max=   0.000, mean=   0.000, NaNs=0
  north : min=   0.000, max=   0.000, mean=   0.000, NaNs=0
  west  : min=   0.000, max=   0.000, mean=   0.000, NaNs=0
  east  : min=   0.000, max=   0.000, mean=   0.000, NaNs=0

wind_v boundary statistics:
  south : min=   0.000, max=   0.000, mean=   0.000, NaNs=0
  north : min=   0.000, max=   0.000, mean=   0.000, NaNs=0
  west  : min=   0.000, max=   0.000, mean=   0.000, NaNs=0
  east  : min=   0.000, max=   0.000, mean=   0.000, NaNs=0

------------------------------------------------------------------------------------------
16. DERIVATIVE READINESS
------------------------------------------------------------------------------------------
  Required variables                 : PASS
  Latitude grid                      : PASS
  Longitude grid                     : PASS
  No infinite U/V values             : PASS
  U/V missing masks identical        : PASS
  Wind-speed consistency             : PASS

------------------------------------------------------------------------------------------
17. AUTOMATED RECOMMENDATION
------------------------------------------------------------------------------------------
Permanent NaN spatial cells: 0 / 24,341

WARNING:
  No permanently masked spatial cells were detected.
  Do NOT assume the domain is ocean-only.
  The wind dataset appears fully populated spatially.
  A land-sea mask may therefore be required before calculating stress curl.

U/V contain no NaNs.
  Therefore land is NOT explicitly represented as NaN in the current wind dataset.
  Do not differentiate across the raw grid blindly.

NEXT STEP:
  Review this diagnostic output.
  If land masking is required, establish the mask first.
  Then implement wind-stress calculation and geographic curl using latitude-dependent dx and constant dy.

==========================================================================================
DIAGNOSTIC COMPLETE
==========================================================================================