from pathlib import Path
import copernicusmarine

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
VARIABLES = ["thetao", "so"]

START = "2021-01-01"
END   = "2024-11-19"

# Same Phase 2 domain
LON_MIN = 45.0
LON_MAX = 105.0
LAT_MIN = 5.0
LAT_MAX = 30.0

# Narrow range around the native GLORYS level ~1062.44 m
MIN_DEPTH = 1050.0
MAX_DEPTH = 1070.0

OUTPUT_DIR = Path("data/raw/glorys")
OUTPUT_FILE = "glorys_deep_1062.nc"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("OceanEmbed — GLORYS Deep-Level Download")
print("=" * 90)
print(f"Dataset : {DATASET_ID}")
print(f"Variables: {VARIABLES}")
print(f"Time    : {START} -> {END}")
print(f"Domain  : {LAT_MIN}N-{LAT_MAX}N, {LON_MIN}E-{LON_MAX}E")
print(f"Depth   : {MIN_DEPTH} -> {MAX_DEPTH} m")
print(f"Output  : {OUTPUT_DIR / OUTPUT_FILE}")
print("=" * 90)

copernicusmarine.subset(
    dataset_id=DATASET_ID,
    variables=VARIABLES,

    minimum_longitude=LON_MIN,
    maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN,
    maximum_latitude=LAT_MAX,

    start_datetime=f"{START}T00:00:00",
    end_datetime=f"{END}T23:59:59",

    minimum_depth=MIN_DEPTH,
    maximum_depth=MAX_DEPTH,

    output_directory=str(OUTPUT_DIR),
    output_filename=OUTPUT_FILE,

    file_format="netcdf",
    skip_existing=True,
    netcdf_compression_level=4,
)

print("\nDOWNLOAD COMPLETE")
print(f"Created: {OUTPUT_DIR / OUTPUT_FILE}")
