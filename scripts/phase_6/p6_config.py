from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE5_DIR = PROJECT_ROOT / "data" / "processed" / "phase5"
PHASE6_DIR = PROJECT_ROOT / "data" / "processed" / "phase6"


INPUT_TENSOR_PATH = PHASE5_DIR / "input_tensor.nc"
OCEAN_MASK_PATH = PHASE5_DIR / "ocean_mask.nc"

NORMALIZATION_STATS_PATH = PHASE6_DIR / "normalization_stats.json"
NORMALIZED_TENSOR_PATH = PHASE6_DIR / "input_tensor_normalized.nc"

# expected tensor geometry
EXPECTED_TIME_SIZE = 1419
EXPECTED_LATITUDE_SIZE = 101
EXPECTED_LONGITUDE_SIZE = 241
EXPECTED_CHANNEL_SIZE = 12
EXPECTED_SHAPE = (
    EXPECTED_TIME_SIZE,
    EXPECTED_LATITUDE_SIZE,
    EXPECTED_LONGITUDE_SIZE,
    EXPECTED_CHANNEL_SIZE,
)

# frozen channel contract
CHANNELS = [
    "SST",
    "SSS",
    "SSHA",
    "wind_U",
    "wind_V",
    "current_U",
    "current_V",
    "wind_stress_curl",
    "latitude",
    "longitude",
    "sin_day_of_year",
    "cos_day_of_year",
]

NORMALIZE_CHANNEL_INDICES = tuple(range(10))
UNCHANGED_CHANNEL_INDICES = (10, 11)

# Chronological training cutoff.
# Total timestamps: 1419, Training: 993,
TRAINING_SIZE = 993
TRAINING_SLICE  = slice(0, TRAINING_SIZE)

# spatial mask contract
OCEAN_MASK_VALUE = 1

# num safety. stdev below considered invalid. no replacement with 1.0
STD_MINIMUM = 0.0

# expected normalization method
NORMALIZATION_METHOD = "z-score"
NORMALIZATION_FORMULA = "(x - mean) / std"
