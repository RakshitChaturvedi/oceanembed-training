==========================================================================================
WIND / SST OCEAN MASK DIAGNOSTIC
==========================================================================================
SST : /home/rakshitchaturvedi/Desktop/Projects/oceanembed-training/data/processed/phase2/sst.nc
WIND: /home/rakshitchaturvedi/Desktop/Projects/oceanembed-training/data/processed/phase2/wind.nc

Datasets opened successfully.

SST variables : ['sst']
Wind variables: ['wind_u', 'wind_v', 'wind_speed']

==========================================================================================
1. GRID ALIGNMENT
==========================================================================================
time      : PASS
latitude  : PASS
longitude : PASS

==========================================================================================
2. SST PERMANENTLY MISSING MASK
==========================================================================================

SST permanently NaN cells
  Count:      12,925
  Percentage: 53.0997%
  Latitude:   5.00 -> 30.00
  Longitude:  45.00 -> 105.00

SST cells with at least one valid observation
  Count:      11,416
  Percentage: 46.9003%
  Latitude:   5.25 -> 29.75
  Longitude:  45.25 -> 104.75

==========================================================================================
3. WIND PERMANENT ZERO MASK
==========================================================================================

Wind cells permanently (U,V) = (0,0)
  Count:      680
  Percentage: 2.7936%
  Latitude:   5.00 -> 30.00
  Longitude:  45.00 -> 105.00

Wind cells with at least one nonzero observation
  Count:      23,661
  Percentage: 97.2064%
  Latitude:   5.25 -> 29.75
  Longitude:  45.25 -> 104.75

==========================================================================================
4. SST / WIND MASK OVERLAP
==========================================================================================

A. Wind permanently zero AND SST permanently NaN
  Count:      680
  Percentage: 2.7936%
  Latitude:   5.00 -> 30.00
  Longitude:  45.00 -> 105.00

B. Wind permanently zero BUT SST has valid data
  Count:      0
  Percentage: 0.0000%

C. SST permanently NaN BUT wind is not permanently zero
  Count:      12,245
  Percentage: 50.3061%
  Latitude:   5.25 -> 29.75
  Longitude:  45.25 -> 104.75

==========================================================================================
5. MASK AGREEMENT
==========================================================================================
Wind permanently zero cells : 680
SST permanently NaN cells   : 12,925
Overlap                     : 680

Of permanently-zero wind cells, 100.00% are permanently NaN in SST.
Of permanently-NaN SST cells, 5.26% are permanently-zero in wind.

==========================================================================================
6. SPATIAL COHERENCE OF WIND-ZERO MASK
==========================================================================================
Wind-zero cells:        680
Clustered cells:        680
Isolated cells:         0
Clustered fraction:     100.00%
Isolated fraction:      0.00%

==========================================================================================
7. NON-OVERLAPPING MASK COORDINATES
==========================================================================================
No cells found where wind is permanently zero but SST is not permanently NaN.

Found 12,245 cells where SST is permanently NaN but wind is not permanently zero.

First 30:
  lat=  5.25, lon=  45.25
  lat=  5.25, lon=  45.50
  lat=  5.25, lon=  45.75
  lat=  5.25, lon=  46.00
  lat=  5.25, lon=  46.25
  lat=  5.25, lon=  46.50
  lat=  5.25, lon=  46.75
  lat=  5.25, lon=  47.00
  lat=  5.25, lon=  47.25
  lat=  5.25, lon=  47.50
  lat=  5.25, lon=  47.75
  lat=  5.25, lon=  48.00
  lat=  5.25, lon=  48.25
  lat=  5.25, lon=  48.50
  lat=  5.25, lon=  95.25
  lat=  5.25, lon=  95.50
  lat=  5.25, lon=  95.75
  lat=  5.25, lon=  96.00
  lat=  5.25, lon=  96.25
  lat=  5.25, lon=  96.50
  lat=  5.25, lon=  96.75
  lat=  5.25, lon=  97.00
  lat=  5.25, lon=  97.25
  lat=  5.25, lon=  97.50
  lat=  5.25, lon= 100.25
  lat=  5.25, lon= 100.50
  lat=  5.25, lon= 100.75
  lat=  5.25, lon= 101.00
  lat=  5.25, lon= 101.25
  lat=  5.25, lon= 101.50

==========================================================================================
8. TEMPORAL WIND ZERO DIAGNOSTIC
==========================================================================================

Cells that are (0,0) on at least one day
  Count:      680
  Percentage: 2.7936%
  Latitude:   5.00 -> 30.00
  Longitude:  45.00 -> 105.00

Cells intermittently (0,0)
  Count:      0
  Percentage: 0.0000%

==========================================================================================
9. RAW WIND NaN / INF CHECK
==========================================================================================

wind_u
  NaNs: 0
  Infs: 0

wind_v
  NaNs: 0
  Infs: 0

==========================================================================================
10. AUTOMATED RECOMMENDATION
==========================================================================================
SAFE TO PROCEED WITH MASKING.
The permanently-zero wind cells correspond to permanently-missing SST cells.
Use the permanent SST-NaN mask as the ocean/land exclusion mask before calculating spatial derivatives.

==========================================================================================
DIAGNOSTIC COMPLETE
==========================================================================================
