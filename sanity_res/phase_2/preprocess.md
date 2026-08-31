====================================================================================================
FILE: sst.nc
PATH: /home/rakshitchaturvedi/Desktop/Projects/oceanembed-training/data/processed/phase2/sst.nc
====================================================================================================

DATASET:
<xarray.Dataset> Size: 276MB
Dimensions:    (time: 1419, latitude: 101, longitude: 241)
Coordinates:
  * time       (time) datetime64[ns] 11kB 2021-01-01 2021-01-02 ... 2024-11-19
  * latitude   (latitude) float64 808B 5.0 5.25 5.5 5.75 ... 29.5 29.75 30.0
  * longitude  (longitude) float64 2kB 45.0 45.25 45.5 ... 104.5 104.8 105.0
Data variables:
    sst        (time, latitude, longitude) float64 276MB ...
Attributes: (12/55)
    Conventions:                     CF-1.6, ACDD-1.3
    title:                           NOAA/NCEI 1/4 Degree Daily Optimum Inter...
    id:                              NCEI-L4_GHRSST-SSTblend-AVHRR_OI
    references:                      Reynolds, et al.(2009) What is New in Ve...
    institution:                     NOAA/NESDIS/NCEI
    creator_name:                    NCEI Products and Services
    ...                              ...
    oceanembed_regrid_method:        bilinear
    oceanembed_grid_resolution:      0.25 degrees
    oceanembed_phase:                phase2
    oceanembed_temporal_resolution:  daily
    oceanembed_spatial_domain:       5.0N-30.0N, 45.0E-105.0E
    oceanembed_spatial_resolution:   0.25deg

[TIME]
Start: 2021-01-01 00:00:00
End:   2024-11-19 00:00:00
Count: 1419
Duplicates: 0

Time intervals:
1 days    1418
Name: count, dtype: int64

Missing dates: 0
Extra dates: 0

Time validation:
  Count:        PASS
  Range:        PASS
  Duplicates:   PASS
  Missing:      PASS
  Extra:        PASS

[LATITUDE]
First: 5.0
Last:  30.0
Count: 101
Spacing: {'min': 0.25, 'max': 0.25, 'unique': array([0.25])}

[LONGITUDE]
First: 45.0
Last:  105.0
Count: 241
Spacing: {'min': 0.25, 'max': 0.25, 'unique': array([0.25])}

Horizontal grid validation:
  Latitude:   PASS
  Longitude:  PASS

[VARIABLES]
Expected: ['sst']
Actual:   ['sst']
Variable set: PASS

--------------------------------------------------------------------------------
sst
  dims: ('time', 'latitude', 'longitude')
  shape: (1419, 101, 241)
  dtype: float64
  units: kelvin
  long_name: analysed sea surface temperature

[MISSING VALUE DIAGNOSTICS]

sst
  Total:       34539879
  NaNs:        17375655
  NaN %:       50.3061
  Infs:        0
  Finite:      17164224
  NaN spatial distribution:
    Always NaN:    12245 (50.31%)
    Never NaN:     12096 (49.69%)
    Sometimes NaN: 0 (0.00%)

[PHYSICAL SANITY]

sst
  Allowed range: [200.0, 330.0]
  Actual min:    0.0
  Actual max:    309.5149841308594
  NaNs:          17375655
  Infs:          0
  Below range:   964920
  Above range:   0
  Result:        FAIL

[SST MASK / ZERO DIAGNOSTICS]
Zero values: 964920
Zero %:      2.793640%

Interpretation:
  0 K is physically invalid for ocean SST.
  These values should be treated as a masking/
  fill-value issue rather than legitimate SST.

----------------------------------------------------------------------------------------------------
sst.nc FINAL RESULT: FAIL
----------------------------------------------------------------------------------------------------

===================================================================================================
FILE: sss.nc
PATH: /home/rakshitchaturvedi/Desktop/Projects/oceanembed-training/data/processed/phase2/sss.nc
====================================================================================================

DATASET:
<xarray.Dataset> Size: 276MB
Dimensions:    (time: 1419, latitude: 101, longitude: 241)
Coordinates:
  * time       (time) datetime64[ns] 11kB 2021-01-01 2021-01-02 ... 2024-11-19
  * latitude   (latitude) float64 808B 5.0 5.25 5.5 5.75 ... 29.5 29.75 30.0
  * longitude  (longitude) float64 2kB 45.0 45.25 45.5 ... 104.5 104.8 105.0
Data variables:
    sss        (time, latitude, longitude) float64 276MB ...
Attributes: (12/51)
    Conventions:                     CF-1.8, ACDD-1.3
    standard_name_vocabulary:        CF Standard Name Table v27
    Title:                           Multi-Mission Optimally Interpolated Sea...
    Short_Name:                      OISSS_L4_multimission_7d_v2
    Version:                         V2.0
    Processing_Level:                Level 4
    ...                              ...
    oceanembed_regrid_method:        bilinear
    oceanembed_grid_resolution:      0.25 degrees
    oceanembed_phase:                phase2
    oceanembed_temporal_resolution:  daily
    oceanembed_spatial_domain:       5.0N-30.0N, 45.0E-105.0E
    oceanembed_spatial_resolution:   0.25deg

[TIME]
Start: 2021-01-01 00:00:00
End:   2024-11-19 00:00:00
Count: 1419
Duplicates: 0

Time intervals:
1 days    1418
Name: count, dtype: int64

Missing dates: 0
Extra dates: 0

Time validation:
  Count:        PASS
  Range:        PASS
  Duplicates:   PASS
  Missing:      PASS
  Extra:        PASS

[LATITUDE]
First: 5.0
Last:  30.0
Count: 101
Spacing: {'min': 0.25, 'max': 0.25, 'unique': array([0.25])}

[LONGITUDE]
First: 45.0
Last:  105.0
Count: 241
Spacing: {'min': 0.25, 'max': 0.25, 'unique': array([0.25])}

Horizontal grid validation:
  Latitude:   PASS
  Longitude:  PASS

[VARIABLES]
Expected: ['sss']
Actual:   ['sss']
Variable set: PASS

--------------------------------------------------------------------------------
sss
  dims: ('time', 'latitude', 'longitude')
  shape: (1419, 101, 241)
  dtype: float64
  units: 1e-3
  long_name: sea surface salinity

[MISSING VALUE DIAGNOSTICS]

sss
  Total:       34539879
  NaNs:        18888689
  NaN %:       54.6866
  Infs:        0
  Finite:      15651190
  NaN spatial distribution:
    Always NaN:    13188 (54.18%)
    Never NaN:     10296 (42.30%)
    Sometimes NaN: 857 (3.52%)

[PHYSICAL SANITY]

sss
  Allowed range: [0.0, 50.0]
  Actual min:    0.0
  Actual max:    41.75785533149808
  NaNs:          18888689
  Infs:          0
  Below range:   0
  Above range:   0
  Result:        PASS

----------------------------------------------------------------------------------------------------
sss.nc FINAL RESULT: PASS
----------------------------------------------------------------------------------------------------
===================================================================================================
FILE: ssh.nc
PATH: /home/rakshitchaturvedi/Desktop/Projects/oceanembed-training/data/processed/phase2/ssh.nc
====================================================================================================

DATASET:
<xarray.Dataset> Size: 276MB
Dimensions:    (time: 1419, latitude: 101, longitude: 241)
Coordinates:
  * time       (time) datetime64[ns] 11kB 2021-01-01 2021-01-02 ... 2024-11-19
  * latitude   (latitude) float64 808B 5.0 5.25 5.5 5.75 ... 29.5 29.75 30.0
  * longitude  (longitude) float64 2kB 45.0 45.25 45.5 ... 104.5 104.8 105.0
Data variables:
    ssh        (time, latitude, longitude) float64 276MB ...
Attributes: (12/52)
    Conventions:                     CF-1.6
    Metadata_Conventions:            Unidata Dataset Discovery v1.0
    cdm_data_type:                   Grid
    comment:                         Sea Surface Height measured by Altimetry...
    contact:                         servicedesk.cmems@mercator-ocean.eu
    creator_email:                   servicedesk.cmems@mercator-ocean.eu
    ...                              ...
    oceanembed_regrid_method:        bilinear
    oceanembed_grid_resolution:      0.25 degrees
    oceanembed_phase:                phase2
    oceanembed_temporal_resolution:  daily
    oceanembed_spatial_domain:       5.0N-30.0N, 45.0E-105.0E
    oceanembed_spatial_resolution:   0.25deg

[TIME]
Start: 2021-01-01 00:00:00
End:   2024-11-19 00:00:00
Count: 1419
Duplicates: 0

Time intervals:
1 days    1418
Name: count, dtype: int64

Missing dates: 0
Extra dates: 0

Time validation:
  Count:        PASS
  Range:        PASS
  Duplicates:   PASS
  Missing:      PASS
  Extra:        PASS

[LATITUDE]
First: 5.0
Last:  30.0
Count: 101
Spacing: {'min': 0.25, 'max': 0.25, 'unique': array([0.25])}

[LONGITUDE]
First: 45.0
Last:  105.0
Count: 241
Spacing: {'min': 0.25, 'max': 0.25, 'unique': array([0.25])}

Horizontal grid validation:
  Latitude:   PASS
  Longitude:  PASS

[VARIABLES]
Expected: ['ssh']
Actual:   ['ssh']
Variable set: PASS

--------------------------------------------------------------------------------
ssh
  dims: ('time', 'latitude', 'longitude')
  shape: (1419, 101, 241)
  dtype: float64
  units: m
  long_name: Sea level anomaly

[MISSING VALUE DIAGNOSTICS]

ssh
  Total:       34539879
  NaNs:        16826502
  NaN %:       48.7162
  Infs:        0
  Finite:      17713377
  NaN spatial distribution:
    Always NaN:    11858 (48.72%)
    Never NaN:     12483 (51.28%)
    Sometimes NaN: 0 (0.00%)

[PHYSICAL SANITY]

ssh
  Allowed range: [-5.0, 5.0]
  Actual min:    -0.6032013450141034
  Actual max:    0.898396540306781
  NaNs:          16826502
  Infs:          0
  Below range:   0
  Above range:   0
  Result:        PASS

----------------------------------------------------------------------------------------------------
ssh.nc FINAL RESULT: PASS
----------------------------------------------------------------------------------------------------
==================================================================================================
FILE: wind.nc
PATH: /home/rakshitchaturvedi/Desktop/Projects/oceanembed-training/data/processed/phase2/wind.nc
====================================================================================================

DATASET:
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

[TIME]
Start: 2021-01-01 00:00:00
End:   2024-11-19 00:00:00
Count: 1419
Duplicates: 0

Time intervals:
1 days    1418
Name: count, dtype: int64

Missing dates: 0
Extra dates: 0

Time validation:
  Count:        PASS
  Range:        PASS
  Duplicates:   PASS
  Missing:      PASS
  Extra:        PASS

[LATITUDE]
First: 5.0
Last:  30.0
Count: 101
Spacing: {'min': 0.25, 'max': 0.25, 'unique': array([0.25])}

[LONGITUDE]
First: 45.0
Last:  105.0
Count: 241
Spacing: {'min': 0.25, 'max': 0.25, 'unique': array([0.25])}

Horizontal grid validation:
  Latitude:   PASS
  Longitude:  PASS

[VARIABLES]
Expected: ['wind_u', 'wind_v', 'wind_speed']
Actual:   ['wind_u', 'wind_v', 'wind_speed']
Variable set: PASS

--------------------------------------------------------------------------------
wind_u
  dims: ('time', 'latitude', 'longitude')
  shape: (1419, 101, 241)
  dtype: float32
  units: m s-1
  long_name: u-wind vector component at 10 meters

--------------------------------------------------------------------------------
wind_v
  dims: ('time', 'latitude', 'longitude')
  shape: (1419, 101, 241)
  dtype: float32
  units: m s-1
  long_name: v-wind vector component at 10 meters

--------------------------------------------------------------------------------
wind_speed
  dims: ('time', 'latitude', 'longitude')
  shape: (1419, 101, 241)
  dtype: float32
  units: m s-1
  long_name: Wind speed derived from daily mean components

[MISSING VALUE DIAGNOSTICS]

wind_u
  Total:       34539879
  NaNs:        0
  NaN %:       0.0000
  Infs:        0
  Finite:      34539879
  NaN spatial distribution:
    Always NaN:    0 (0.00%)
    Never NaN:     24341 (100.00%)
    Sometimes NaN: 0 (0.00%)

wind_v
  Total:       34539879
  NaNs:        0
  NaN %:       0.0000
  Infs:        0
  Finite:      34539879
  NaN spatial distribution:
    Always NaN:    0 (0.00%)
    Never NaN:     24341 (100.00%)
    Sometimes NaN: 0 (0.00%)

wind_speed
  Total:       34539879
  NaNs:        0
  NaN %:       0.0000
  Infs:        0
  Finite:      34539879
  NaN spatial distribution:
    Always NaN:    0 (0.00%)
    Never NaN:     24341 (100.00%)
    Sometimes NaN: 0 (0.00%)

[PHYSICAL SANITY]

wind_u
  Allowed range: [-100.0, 100.0]
  Actual min:    -20.150171279907227
  Actual max:    26.10869598388672
  NaNs:          0
  Infs:          0
  Below range:   0
  Above range:   0
  Result:        PASS

wind_v
  Allowed range: [-100.0, 100.0]
  Actual min:    -21.52005958557129
  Actual max:    26.70372772216797
  NaNs:          0
  Infs:          0
  Below range:   0
  Above range:   0
  Result:        PASS

wind_speed
  Allowed range: [0.0, 100.0]
  Actual min:    0.0
  Actual max:    28.337739944458008
  NaNs:          0
  Infs:          0
  Below range:   0
  Above range:   0
  Result:        PASS

[WIND CONSISTENCY]
Formula:
  wind_speed = sqrt(wind_u² + wind_v²)

Maximum absolute error: 1.8160863497485025e-06
Mean absolute error:    9.387173044744518e-08
Valid points:            34539879
Result:                  PASS

----------------------------------------------------------------------------------------------------
wind.nc FINAL RESULT: PASS
----------------------------------------------------------------------------------------------------
===================================================================================================
FILE: currents.nc
PATH: /home/rakshitchaturvedi/Desktop/Projects/oceanembed-training/data/processed/phase2/currents.nc
====================================================================================================

DATASET:
<xarray.Dataset> Size: 553MB
Dimensions:    (time: 1419, latitude: 101, longitude: 241)
Coordinates:
  * time       (time) datetime64[ns] 11kB 2021-01-01 2021-01-02 ... 2024-11-19
  * latitude   (latitude) float64 808B 5.0 5.25 5.5 5.75 ... 29.5 29.75 30.0
  * longitude  (longitude) float64 2kB 45.0 45.25 45.5 ... 104.5 104.8 105.0
Data variables:
    current_u  (time, latitude, longitude) float64 276MB ...
    current_v  (time, latitude, longitude) float64 276MB ...
Attributes: (12/45)
    title:                           Ocean Surface Current Analyses Real-time...
    summary:                         Global, daily, 0.25 degree geostrophic a...
    keywords:                        ocean currents,ocean circulation,surface...
    Conventions:                     CF-1.8 Standard Names v77, ACDD-1.3, net...
    id:                              OSCAR_L4_OC_FINAL_V2.0
    history:                         OSCAR 0.25 degree daily version 2.0 repl...
    ...                              ...
    oceanembed_regrid_method:        bilinear
    oceanembed_grid_resolution:      0.25 degrees
    oceanembed_phase:                phase2
    oceanembed_temporal_resolution:  daily
    oceanembed_spatial_domain:       5.0N-30.0N, 45.0E-105.0E
    oceanembed_spatial_resolution:   0.25deg

[TIME]
Start: 2021-01-01 00:00:00
End:   2024-11-19 00:00:00
Count: 1419
Duplicates: 0

Time intervals:
1 days    1418
Name: count, dtype: int64

Missing dates: 0
Extra dates: 0

Time validation:
  Count:        PASS
  Range:        PASS
  Duplicates:   PASS
  Missing:      PASS
  Extra:        PASS

[LATITUDE]
First: 5.0
Last:  30.0
Count: 101
Spacing: {'min': 0.25, 'max': 0.25, 'unique': array([0.25])}

[LONGITUDE]
First: 45.0
Last:  105.0
Count: 241
Spacing: {'min': 0.25, 'max': 0.25, 'unique': array([0.25])}

Horizontal grid validation:
  Latitude:   PASS
  Longitude:  PASS

[VARIABLES]
Expected: ['current_u', 'current_v']
Actual:   ['current_u', 'current_v']
Variable set: PASS

--------------------------------------------------------------------------------
current_u
  dims: ('time', 'latitude', 'longitude')
  shape: (1419, 101, 241)
  dtype: float64
  units: m s-1
  long_name: zonal total surface current

--------------------------------------------------------------------------------
current_v
  dims: ('time', 'latitude', 'longitude')
  shape: (1419, 101, 241)
  dtype: float64
  units: m s-1
  long_name: meridional total surface current

[MISSING VALUE DIAGNOSTICS]

current_u
  Total:       34539879
  NaNs:        18546535
  NaN %:       53.6960
  Infs:        0
  Finite:      15993344
  NaN spatial distribution:
    Always NaN:    13000 (53.41%)
    Never NaN:     11169 (45.89%)
    Sometimes NaN: 172 (0.71%)

current_v
  Total:       34539879
  NaNs:        18546535
  NaN %:       53.6960
  Infs:        0
  Finite:      15993344
  NaN spatial distribution:
    Always NaN:    13000 (53.41%)
    Never NaN:     11169 (45.89%)
    Sometimes NaN: 172 (0.71%)

[PHYSICAL SANITY]

current_u
  Allowed range: [-10.0, 10.0]
  Actual min:    -2.6521477266322644
  Actual max:    2.706659750808018
  NaNs:          18546535
  Infs:          0
  Below range:   0
  Above range:   0
  Result:        PASS

current_v
  Allowed range: [-10.0, 10.0]
  Actual min:    -1.9702047894812416
  Actual max:    2.3934319967032316
  NaNs:          18546535
  Infs:          0
  Below range:   0
  Above range:   0
  Result:        PASS

----------------------------------------------------------------------------------------------------
currents.nc FINAL RESULT: PASS
----------------------------------------------------------------------------------------------------
====================================================================================================
FILE: armor3d.nc
PATH: /home/rakshitchaturvedi/Desktop/Projects/oceanembed-training/data/processed/phase2/armor3d.nc
====================================================================================================

DATASET:
<xarray.Dataset> Size: 20GB
Dimensions:      (time: 1419, depth: 36, latitude: 101, longitude: 241)
Coordinates:
  * time         (time) datetime64[ns] 11kB 2021-01-01 2021-01-02 ... 2024-11-19
  * depth        (depth) int16 72B 0 5 10 15 20 25 ... 550 600 700 800 900 1000
  * latitude     (latitude) float64 808B 5.0 5.25 5.5 5.75 ... 29.5 29.75 30.0
  * longitude    (longitude) float64 2kB 45.0 45.25 45.5 ... 104.5 104.8 105.0
Data variables:
    temperature  (time, depth, latitude, longitude) float64 10GB ...
    salinity     (time, depth, latitude, longitude) float64 10GB ...
Attributes: (12/20)
    Conventions:                     CF-1.0
    description:                     ARMOR3D REP Copernicus Marine Service No...
    domain_name:                     GLO
    history:                         2025-10-11 04:04:02 ARMOR3D REP - TSHUV ...
    institution:                     CLS
    lat_max:                         89.9375
    ...                              ...
    oceanembed_regrid_method:        bilinear
    oceanembed_grid_resolution:      0.25 degrees
    oceanembed_phase:                phase2
    oceanembed_temporal_resolution:  daily
    oceanembed_spatial_domain:       5.0N-30.0N, 45.0E-105.0E
    oceanembed_spatial_resolution:   0.25deg

[TIME]
Start: 2021-01-01 00:00:00
End:   2024-11-19 00:00:00
Count: 1419
Duplicates: 0

Time intervals:
1 days    1418
Name: count, dtype: int64

Missing dates: 0
Extra dates: 0

Time validation:
  Count:        PASS
  Range:        PASS
  Duplicates:   PASS
  Missing:      PASS
  Extra:        PASS

[LATITUDE]
First: 5.0
Last:  30.0
Count: 101
Spacing: {'min': 0.25, 'max': 0.25, 'unique': array([0.25])}

[LONGITUDE]
First: 45.0
Last:  105.0
Count: 241
Spacing: {'min': 0.25, 'max': 0.25, 'unique': array([0.25])}

Horizontal grid validation:
  Latitude:   PASS
  Longitude:  PASS

[VARIABLES]
Expected: ['temperature', 'salinity']
Actual:   ['temperature', 'salinity']
Variable set: PASS

--------------------------------------------------------------------------------
temperature
  dims: ('time', 'depth', 'latitude', 'longitude')
  shape: (1419, 36, 101, 241)
  dtype: float64
  units: degrees_C
  long_name: temperature

--------------------------------------------------------------------------------
salinity
  dims: ('time', 'depth', 'latitude', 'longitude')
  shape: (1419, 36, 101, 241)
  dtype: float64
  units: 0.001
  long_name: salinity

[MISSING VALUE DIAGNOSTICS]

temperature
  Total:       1243435644
  NaNs:        709990974
  NaN %:       57.0991
  Infs:        0
  Finite:      533444670
  NaN spatial distribution:
    Always NaN:    500346 (57.10%)
    Never NaN:     375930 (42.90%)
    Sometimes NaN: 0 (0.00%)

salinity
  Total:       1243435644
  NaNs:        709990974
  NaN %:       57.0991
  Infs:        0
  Finite:      533444670
  NaN spatial distribution:
    Always NaN:    500346 (57.10%)
    Never NaN:     375930 (42.90%)
    Sometimes NaN: 0 (0.00%)

[PHYSICAL SANITY]

temperature
  Allowed range: [-5.0, 50.0]
  Actual min:    0.0
  Actual max:    37.445753473422
  NaNs:          709990974
  Infs:          0
  Below range:   0
  Above range:   0
  Result:        PASS

salinity
  Allowed range: [0.0, 50.0]
  Actual min:    0.0
  Actual max:    42.50574717004335
  NaNs:          709990974
  Infs:          0
  Below range:   0
  Above range:   0
  Result:        PASS

[SUBSURFACE VALIDATION]

Required dimensions:
  {'latitude', 'longitude', 'depth', 'time'}
Result: PASS

Time:
  Count: 1419
  Start: 2021-01-01 00:00:00
  End:   2024-11-19 00:00:00
  Result: PASS

Depth:
  Count: 36
  First: 0.0
  Last:  1000.0
  Finite:              PASS
  Strictly increasing: PASS
  Non-negative:        PASS
  Levels:
    [   0.    5.   10.   15.   20.   25.   30.   35.   40.   45.   50.   55.
   60.   65.   70.   80.   90.  100.  125.  150.  175.  200.  225.  250.
  275.  300.  350.  400.  450.  500.  550.  600.  700.  800.  900. 1000.]
  Result: PASS

Depth spacing:
  Min spacing: 5.0
  Max spacing: 100.0

Variable dimensions:
  temperature: ('time', 'depth', 'latitude', 'longitude') -> PASS
  salinity: ('time', 'depth', 'latitude', 'longitude') -> PASS

Horizontal grid:
  Latitude: PASS
  Longitude: PASS

[DEPTH COVERAGE]

temperature
  Surface depth coverage: 49.64%
  Deepest depth coverage: 38.57%
  Completely empty levels: 0
  Result: PASS

salinity
  Surface depth coverage: 49.64%
  Deepest depth coverage: 38.57%
  Completely empty levels: 0
  Result: PASS

[SUBSURFACE PHYSICAL SANITY]

temperature
  Allowed range: [-5.0, 50.0]
  Actual min:    0.0
  Actual max:    37.445753473422
  NaNs:          709990974
  Infs:          0
  Below range:   0
  Above range:   0
  Result:        PASS

salinity
  Allowed range: [0.0, 50.0]
  Actual min:    0.0
  Actual max:    42.50574717004335
  NaNs:          709990974
  Infs:          0
  Below range:   0
  Above range:   0
  Result:        PASS

[DEPTH-WISE PHYSICAL COVERAGE]

temperature
  Minimum valid physical coverage: 38.57%
  Maximum valid physical coverage: 49.64%
  Levels with zero valid values: 0
  Result: PASS

salinity
  Minimum valid physical coverage: 38.57%
  Maximum valid physical coverage: 49.64%
  Levels with zero valid values: 0
  Result: PASS

Subsurface validation result:
  PASS

----------------------------------------------------------------------------------------------------
armor3d.nc FINAL RESULT: PASS
----------------------------------------------------------------------------------------------------
===================================================================================================
FILE: glorys.nc
PATH: /home/rakshitchaturvedi/Desktop/Projects/oceanembed-training/data/processed/phase2/glorys.nc
====================================================================================================

DATASET:
<xarray.Dataset> Size: 19GB
Dimensions:      (time: 1419, depth: 35, latitude: 101, longitude: 241)
Coordinates:
  * time         (time) datetime64[ns] 11kB 2021-01-01 2021-01-02 ... 2024-11-19
  * depth        (depth) float32 140B 0.494 1.541 2.646 ... 643.6 763.3 902.3
  * latitude     (latitude) float64 808B 5.0 5.25 5.5 5.75 ... 29.5 29.75 30.0
  * longitude    (longitude) float64 2kB 45.0 45.25 45.5 ... 104.5 104.8 105.0
Data variables:
    temperature  (time, depth, latitude, longitude) float64 10GB ...
    salinity     (time, depth, latitude, longitude) float64 10GB ...
Attributes: (12/34)
    Conventions:                     CF-1.4
    bulletin_date:                   2021-07-07 00:00:00
    bulletin_type:                   operational
    comment:                         CMEMS product
    domain_name:                     GL12
    easting:                         longitude
    ...                              ...
    oceanembed_regrid_method:        bilinear
    oceanembed_grid_resolution:      0.25 degrees
    oceanembed_phase:                phase2
    oceanembed_temporal_resolution:  daily
    oceanembed_spatial_domain:       5.0N-30.0N, 45.0E-105.0E
    oceanembed_spatial_resolution:   0.25deg

[TIME]
Start: 2021-01-01 00:00:00
End:   2024-11-19 00:00:00
Count: 1419
Duplicates: 0

Time intervals:
1 days    1418
Name: count, dtype: int64

Missing dates: 0
Extra dates: 0

Time validation:
  Count:        PASS
  Range:        PASS
  Duplicates:   PASS
  Missing:      PASS
  Extra:        PASS

[LATITUDE]
First: 5.0
Last:  30.0
Count: 101
Spacing: {'min': 0.25, 'max': 0.25, 'unique': array([0.25])}

[LONGITUDE]
First: 45.0
Last:  105.0
Count: 241
Spacing: {'min': 0.25, 'max': 0.25, 'unique': array([0.25])}

Horizontal grid validation:
  Latitude:   PASS
  Longitude:  PASS

[VARIABLES]
Expected: ['temperature', 'salinity']
Actual:   ['temperature', 'salinity']
Variable set: PASS

--------------------------------------------------------------------------------
temperature
  dims: ('time', 'depth', 'latitude', 'longitude')
  shape: (1419, 35, 101, 241)
  dtype: float64
  units: degrees_C
  long_name: Temperature

--------------------------------------------------------------------------------
salinity
  dims: ('time', 'depth', 'latitude', 'longitude')
  shape: (1419, 35, 101, 241)
  dtype: float64
  units: 1e-3
  long_name: Salinity

[MISSING VALUE DIAGNOSTICS]

temperature
  Total:       1208895765
  NaNs:        685344363
  NaN %:       56.6918
  Infs:        0
  Finite:      523551402
  NaN spatial distribution:
    Always NaN:    482977 (56.69%)
    Never NaN:     368958 (43.31%)
    Sometimes NaN: 0 (0.00%)

salinity
  Total:       1208895765
  NaNs:        685344363
  NaN %:       56.6918
  Infs:        0
  Finite:      523551402
  NaN spatial distribution:
    Always NaN:    482977 (56.69%)
    Never NaN:     368958 (43.31%)
    Sometimes NaN: 0 (0.00%)

[PHYSICAL SANITY]

temperature
  Allowed range: [-5.0, 50.0]
  Actual min:    5.051026962697506
  Actual max:    36.98852502554655
  NaNs:          685344363
  Infs:          0
  Below range:   0
  Above range:   0
  Result:        PASS

salinity
  Allowed range: [0.0, 50.0]
  Actual min:    1.1093478184193373
  Actual max:    41.03366187773645
  NaNs:          685344363
  Infs:          0
  Below range:   0
  Above range:   0
  Result:        PASS

[SUBSURFACE VALIDATION]

Required dimensions:
  {'latitude', 'longitude', 'depth', 'time'}
Result: PASS

Time:
  Count: 1419
  Start: 2021-01-01 00:00:00
  End:   2024-11-19 00:00:00
  Result: PASS

Depth:
  Count: 35
  First: 0.49402499198913574
  Last:  902.3392944335938
  Finite:              PASS
  Strictly increasing: PASS
  Non-negative:        PASS
  Levels:
    [4.94024992e-01 1.54137504e+00 2.64566898e+00 3.81949496e+00
 5.07822418e+00 6.44061422e+00 7.92956018e+00 9.57299709e+00
 1.14049997e+01 1.34671402e+01 1.58100700e+01 1.84955597e+01
 2.15988197e+01 2.52114105e+01 2.94447308e+01 3.44341507e+01
 4.03440514e+01 4.73736916e+01 5.57642899e+01 6.58072662e+01
 7.78538513e+01 9.23260727e+01 1.09729301e+02 1.30666000e+02
 1.55850693e+02 1.86125595e+02 2.22475204e+02 2.66040314e+02
 3.18127411e+02 3.80213013e+02 4.53937714e+02 5.41088928e+02
 6.43566772e+02 7.63333130e+02 9.02339294e+02]
  Result: PASS

Depth spacing:
  Min spacing: 1.0473500490188599
  Max spacing: 139.00616455078125

Variable dimensions:
  temperature: ('time', 'depth', 'latitude', 'longitude') -> PASS
  salinity: ('time', 'depth', 'latitude', 'longitude') -> PASS

Horizontal grid:
  Latitude: PASS
  Longitude: PASS

[DEPTH COVERAGE]

temperature
  Surface depth coverage: 48.70%
  Deepest depth coverage: 37.39%
  Completely empty levels: 0
  Result: PASS

salinity
  Surface depth coverage: 48.70%
  Deepest depth coverage: 37.39%
  Completely empty levels: 0
  Result: PASS

[SUBSURFACE PHYSICAL SANITY]

temperature
  Allowed range: [-5.0, 50.0]
  Actual min:    5.051026962697506
  Actual max:    36.98852502554655
  NaNs:          685344363
  Infs:          0
  Below range:   0
  Above range:   0
  Result:        PASS

salinity
  Allowed range: [0.0, 50.0]
  Actual min:    1.1093478184193373
  Actual max:    41.03366187773645
  NaNs:          685344363
  Infs:          0
  Below range:   0
  Above range:   0
  Result:        PASS

[DEPTH-WISE PHYSICAL COVERAGE]

temperature
  Minimum valid physical coverage: 37.39%
  Maximum valid physical coverage: 48.70%
  Levels with zero valid values: 0
  Result: PASS

salinity
  Minimum valid physical coverage: 37.39%
  Maximum valid physical coverage: 48.70%
  Levels with zero valid values: 0
  Result: PASS

Subsurface validation result:
  PASS

----------------------------------------------------------------------------------------------------
glorys.nc FINAL RESULT: PASS
----------------------------------------------------------------------------------------------------
===================================================================================================
CROSS-DATASET ALIGNMENT
====================================================================================================

Reference dataset: sst.nc

sst.nc
  Time:       PASS
  Latitude:   PASS
  Longitude:  PASS
  Overall:    PASS

sss.nc
  Time:       PASS
  Latitude:   PASS
  Longitude:  PASS
  Overall:    PASS

ssh.nc
  Time:       PASS
  Latitude:   PASS
  Longitude:  PASS
  Overall:    PASS

wind.nc
  Time:       PASS
  Latitude:   PASS
  Longitude:  PASS
  Overall:    PASS

currents.nc
  Time:       PASS
  Latitude:   PASS
  Longitude:  PASS
  Overall:    PASS

armor3d.nc
  Time:       PASS
  Latitude:   PASS
  Longitude:  PASS
  Overall:    PASS

glorys.nc
  Time:       PASS
  Latitude:   PASS
  Longitude:  PASS
  Overall:    PASS

Cross-dataset alignment result:
  PASS


====================================================================================================
PHASE 2 VALIDATION SUMMARY
====================================================================================================
sst.nc         : FAIL / INCOMPLETE
sss.nc         : PASS
ssh.nc         : PASS
wind.nc        : PASS
currents.nc    : PASS
armor3d.nc     : PASS
glorys.nc      : PASS

Cross-dataset alignment: PASS
====================================================================================================
