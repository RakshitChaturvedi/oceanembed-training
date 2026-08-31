from __future__ import annotations

DOMAIN = {
    "lat_min": 5.0,
    "lat_max": 30.0,
    "lon_min": 45.0,
    "lon_max": 105.0,
    "resolution": 0.25
}

DEPTHS = [0,5,10,20,30,50,75,100,125,150,200,300,500,700,1000]
CHANNEL_ORDER = ["SST", "SSS", "SSHA", "wind_u", "wind_v", "current_u", "current_v",
                 "wind_stress_curl", "lat", "lon", "sin_day_of_year", "cost_day_of_year"]
N_INPUT_CHANNELS = len(CHANNEL_ORDER)
N_OUTPUT_DEPTHS = len(DEPTHS)