==========================================================================================
OCEANEMBED PHASE 4 — WIND STRESS CURL VALIDATION
==========================================================================================

WIND : /home/rakshitchaturvedi/Desktop/Projects/oceanembed-training/data/processed/phase2/wind.nc
SST  : /home/rakshitchaturvedi/Desktop/Projects/oceanembed-training/data/processed/phase2/sst.nc
CURL : /home/rakshitchaturvedi/Desktop/Projects/oceanembed-training/data/processed/phase4/wind_stress_curl.nc

==========================================================================================
1. DATASET STRUCTURE
==========================================================================================

Expected variables:
  ['wind_stress_curl', 'wind_stress_x', 'wind_stress_y']
Actual variables:
  ['wind_stress_curl', 'wind_stress_x', 'wind_stress_y']
Expected variables present                                  : PASS

==========================================================================================
2. GRID ALIGNMENT
==========================================================================================
time                                                        : PASS
latitude                                                    : PASS
longitude                                                   : PASS

==========================================================================================
3. DIMENSIONS
==========================================================================================
time: expected 1419, got 1419                               : PASS
latitude: expected 101, got 101                             : PASS
longitude: expected 241, got 241                            : PASS
================================================================================
4. OCEAN MASK VALIDATION
================================================================================

Expected ocean cells : 11,416
Expected masked cells: 12,925
Curl valid cells     : 10,530
Curl masked cells    : 13,811

Curl values outside SST ocean mask: 0
Curl valid region is fully contained within SST ocean mask  : PASS

SST ocean cells without valid curl: 886
Some SST ocean cells lack curl because of masked neighbors  : EXPECTED

Ocean cells excluded from curl derivative: 886
Expected coastal/edge derivative loss observed  : PASS

==========================================================================================
5. NaN / INF CHECK
==========================================================================================

wind_stress_x
  NaNs: 18,340,575
  Infs: 0
wind_stress_x has no Infs                                   : PASS

wind_stress_y
  NaNs: 18,340,575
  Infs: 0
wind_stress_y has no Infs                                   : PASS

wind_stress_curl
  NaNs: 19,597,809
  Infs: 0
wind_stress_curl has no Infs                                : PASS

==========================================================================================
6. MASKED CELL INTEGRITY
==========================================================================================
Finite curl values inside masked cells: 0
No curl values exist inside masked cells                    : PASS

==========================================================================================
7. WIND STRESS STATISTICS
==========================================================================================

Zonal wind stress [N m-2]
  Min       : -6.63954854e-01
  Max       : 1.12188232e+00
  Mean      : 2.12646481e-02
  Median    : 4.81238915e-03
  Std       : 6.42290562e-02
  P01       : -1.00914955e-01
  P05       : -6.54897392e-02
  P95       : 1.43087938e-01
  P99       : 2.14637816e-01
  P999      : 2.93225229e-01
  Finite    : 16,199,304

Meridional wind stress [N m-2]
  Min       : -7.37519264e-01
  Max       : 1.18313229e+00
  Mean      : 1.09161269e-02
  Median    : -1.47911953e-04
  Std       : 5.96357062e-02
  P01       : -1.11843750e-01
  P05       : -6.94102123e-02
  P95       : 1.27189144e-01
  P99       : 2.03189552e-01
  P999      : 2.96967953e-01
  Finite    : 16,199,304

Max |tau_x|: 1.121882 N/m²
Max |tau_y|: 1.183132 N/m²
Wind stress magnitudes are within broad sanity limits       : PASS

==========================================================================================
8. WIND STRESS CURL STATISTICS
==========================================================================================

Wind stress curl [N m-3]
  Min       : -5.12043060e-06
  Max       : 2.80515080e-05
  Mean      : -1.05114515e-08
  Median    : -2.42789056e-08
  Std       : 2.51334345e-07
  P01       : -5.64957512e-07
  P05       : -3.04825598e-07
  P95       : 3.31642938e-07
  P99       : 7.95631472e-07
  P999      : 1.92247816e-06
  Finite    : 14,942,070

Max |curl| : 2.80515080e-05 N/m³
P99.9 |curl|: 1.99345156e-06 N/m³
Curl magnitude passes broad numerical sanity check          : PASS

==========================================================================================
9. MASK-EDGE / COASTAL SPIKE CHECK
==========================================================================================

Ocean edge cells   : 886
Interior ocean cells: 10,530

Interior |curl|:
  P99   : 9.01276123e-07
  P99.9 : 1.99345156e-06
  Max   : 2.80515080e-05

==========================================================================================
10. DOMAIN BOUNDARY CHECK
==========================================================================================

==========================================================================================
11. TEMPORAL COVERAGE
==========================================================================================

Time count: 1419
Start: 2021-01-01T00:00:00.000000000
End  : 2024-11-19T00:00:00.000000000
Daily time continuity                                       : PASS

==========================================================================================
12. METADATA
==========================================================================================
oceanembed_phase                                            : PASS
oceanembed_feature                                          : PASS
oceanembed_formula                                          : PASS
oceanembed_ocean_mask                                       : PASS
oceanembed_mask_before_derivative                           : PASS
oceanembed_air_density_kg_m3                                : PASS
oceanembed_drag_coefficient                                 : PASS
oceanembed_earth_radius_m                                   : PASS

Mask-before-derivative metadata: true

==========================================================================================
13. FORMULA
==========================================================================================

Recorded formula: curl_tau = d(tau_y)/dx - d(tau_x)/dy
Formula metadata                                            : PASS

==========================================================================================
FINAL VALIDATION RESULT
==========================================================================================

PASS

Phase 4 wind-stress curl passed structural and numerical validation.

Recommended next step: inspect the spatial curl distribution before proceeding to model-input construction.
