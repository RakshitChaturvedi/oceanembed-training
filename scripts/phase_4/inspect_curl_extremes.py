from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(
    "/home/rakshitchaturvedi/Desktop/Projects/oceanembed-training"
)

CURL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase4"
    / "wind_stress_curl.nc"
)
WIND_PATH = Path(
    "/home/rakshitchaturvedi/Desktop/Projects/oceanembed-training/"
    "data/processed/phase2/wind.nc"
)
SST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase2"
    / "sst.nc"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

TOP_N = 20

# Large curl threshold for additional inspection.
# Based on the previous diagnostic, values above ~2e-6 are uncommon.
LARGE_CURL_THRESHOLD = 2.0e-6


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print_header("OCEANEMBED PHASE 4 — CURL EXTREME INSPECTION")

    print(f"CURL: {CURL_PATH}")
    print(f"SST : {SST_PATH}")

    wind = xr.open_dataset(WIND_PATH)
    curl_ds = xr.open_dataset(CURL_PATH)

    u = wind["wind_u"]
    v = wind["wind_v"]
    speed = wind["wind_speed"]

    curl = curl_ds["wind_stress_curl"]
    tau_x = curl_ds["wind_stress_x"]
    tau_y = curl_ds["wind_stress_y"]

    # ------------------------------------------------------------------
    # 1. Find extreme curl locations
    # ------------------------------------------------------------------

    values = np.abs(curl.values)

    flat_indices = np.argsort(
        np.nan_to_num(values, nan=-np.inf).ravel()
    )[-TOP_N:][::-1]

    print("\n" + "=" * 80)
    print(f"TOP {TOP_N} CURL EXTREMES")
    print("=" * 80)

    shape = curl.shape

    for rank, flat_idx in enumerate(flat_indices, start=1):

        t, y, x = np.unravel_index(flat_idx, shape)

        curl_value = float(curl.values[t, y, x])

        if not np.isfinite(curl_value):
            continue

        print(
            f"{rank:2d}. "
            f"time={curl.time.values[t]} "
            f"lat={curl.latitude.values[y]:.2f} "
            f"lon={curl.longitude.values[x]:.2f} "
            f"curl={curl_value:.6e}"
        )

    # ------------------------------------------------------------------
    # 2. Inspect strongest event neighborhood
    # ------------------------------------------------------------------

    max_flat = np.nanargmax(np.abs(curl.values))
    t, y, x = np.unravel_index(max_flat, curl.shape)

    print("\n" + "=" * 80)
    print("STRONGEST EVENT")
    print("=" * 80)

    print(f"Time      : {curl.time.values[t]}")
    print(f"Latitude  : {curl.latitude.values[y]:.2f}")
    print(f"Longitude : {curl.longitude.values[x]:.2f}")
    print(f"Curl      : {curl.values[t, y, x]:.8e}")
    print(f"U         : {u.values[t, y, x]:.6f} m/s")
    print(f"V         : {v.values[t, y, x]:.6f} m/s")
    print(f"Speed     : {speed.values[t, y, x]:.6f} m/s")
    print(f"Tau X     : {tau_x.values[t, y, x]:.8e} N/m²")
    print(f"Tau Y     : {tau_y.values[t, y, x]:.8e} N/m²")

    # ------------------------------------------------------------------
    # 3. Local wind neighborhood
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("LOCAL WIND NEIGHBORHOOD")
    print("=" * 80)

    y0 = max(0, y - 2)
    y1 = min(len(curl.latitude), y + 3)

    x0 = max(0, x - 2)
    x1 = min(len(curl.longitude), x + 3)

    local_speed = speed.isel(
        time=t,
        latitude=slice(y0, y1),
        longitude=slice(x0, x1),
    )

    local_tau_x = tau_x.isel(
        time=t,
        latitude=slice(y0, y1),
        longitude=slice(x0, x1),
    )

    local_tau_y = tau_y.isel(
        time=t,
        latitude=slice(y0, y1),
        longitude=slice(x0, x1),
    )

    print("\nWind speed [m/s]:")
    print(local_speed.values)

    print("\nTau X [N/m²]:")
    print(local_tau_x.values)

    print("\nTau Y [N/m²]:")
    print(local_tau_y.values)

    # ------------------------------------------------------------------
    # 4. Temporal persistence
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEMPORAL PERSISTENCE AT EXTREME LOCATION")
    print("=" * 80)

    ts = curl.isel(
        latitude=y,
        longitude=x,
    )

    nonzero = np.abs(ts.values)

    for idx in np.argsort(
        np.nan_to_num(nonzero, nan=-np.inf)
    )[-15:][::-1]:

        print(
            f"{ts.time.values[idx]} : "
            f"{ts.values[idx]:.8e}"
        )

    # ------------------------------------------------------------------
    # 5. Final interpretation
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print(
        """
Inspect the strongest event using these criteria:

1. Wind speed should be physically plausible.
2. Tau_x/tau_y should increase smoothly with wind speed.
3. Extreme curl should occur across multiple neighboring cells
   rather than one isolated grid cell.
4. The event should have temporal persistence or a coherent
   meteorological structure.
5. Extreme values should not coincide with NaN/mask boundaries.

If these conditions hold, do NOT clip the curl values.
They represent a potentially important physical event.
"""
    )


    with xr.open_dataset(CURL_PATH) as curl_ds, \
         xr.open_dataset(SST_PATH) as sst_ds:

        curl = curl_ds["wind_stress_curl"]
        sst = sst_ds["sst"]

        # ---------------------------------------------------------------------
        # GRID CHECK
        # ---------------------------------------------------------------------

        print_header("1. GRID")

        print(f"Time      : {curl.sizes['time']}")
        print(f"Latitude  : {curl.sizes['latitude']}")
        print(f"Longitude : {curl.sizes['longitude']}")

        # ---------------------------------------------------------------------
        # GLOBAL STATISTICS
        # ---------------------------------------------------------------------

        print_header("2. GLOBAL CURL STATISTICS")

        values = curl.values

        finite_values = values[np.isfinite(values)]

        abs_values = np.abs(finite_values)

        print(f"Finite values : {finite_values.size:,}")

        print(f"Min           : {finite_values.min():.10e}")
        print(f"Max           : {finite_values.max():.10e}")
        print(f"Mean          : {finite_values.mean():.10e}")
        print(f"Std           : {finite_values.std():.10e}")

        print()
        print(f"|curl| P95    : {np.percentile(abs_values, 95):.10e}")
        print(f"|curl| P99    : {np.percentile(abs_values, 99):.10e}")
        print(f"|curl| P99.9  : {np.percentile(abs_values, 99.9):.10e}")
        print(f"|curl| Max    : {abs_values.max():.10e}")

        # ---------------------------------------------------------------------
        # TOP EXTREMES
        # ---------------------------------------------------------------------

        print_header("3. TOP ABSOLUTE CURL VALUES")

        flat_indices = np.flatnonzero(np.isfinite(values))

        sorted_indices = flat_indices[
            np.argsort(np.abs(values.ravel()[flat_indices]))[::-1]
        ]

        top_indices = sorted_indices[:TOP_N]

        times = curl["time"].values
        lats = curl["latitude"].values
        lons = curl["longitude"].values

        rows = []

        for flat_idx in top_indices:

            t_idx, lat_idx, lon_idx = np.unravel_index(
                flat_idx,
                values.shape,
            )

            curl_value = values[
                t_idx,
                lat_idx,
                lon_idx,
            ]

            rows.append(
                {
                    "rank": len(rows) + 1,
                    "curl": curl_value,
                    "abs_curl": abs(curl_value),
                    "time": pd.Timestamp(times[t_idx]),
                    "latitude": lats[lat_idx],
                    "longitude": lons[lon_idx],
                }
            )

        df = pd.DataFrame(rows)

        print(
            df.to_string(
                index=False,
                formatters={
                    "curl": "{:.8e}".format,
                    "abs_curl": "{:.8e}".format,
                },
            )
        )

        # ---------------------------------------------------------------------
        # LARGE VALUE COUNT
        # ---------------------------------------------------------------------

        print_header("4. LARGE CURL COUNT")

        large_mask = np.abs(values) >= LARGE_CURL_THRESHOLD

        large_count = int(large_mask.sum())

        print(
            f"Threshold : |curl| >= "
            f"{LARGE_CURL_THRESHOLD:.3e} N/m³"
        )

        print(f"Count     : {large_count:,}")

        if large_count == 0:
            print("No unusually large curl values found.")

        # ---------------------------------------------------------------------
        # SPATIAL DISTRIBUTION
        # ---------------------------------------------------------------------

        print_header("5. SPATIAL DISTRIBUTION OF EXTREMES")

        if large_count > 0:

            large_indices = np.flatnonzero(large_mask)

            large_rows = []

            for flat_idx in large_indices:

                t_idx, lat_idx, lon_idx = np.unravel_index(
                    flat_idx,
                    values.shape,
                )

                large_rows.append(
                    {
                        "time": pd.Timestamp(times[t_idx]),
                        "latitude": float(lats[lat_idx]),
                        "longitude": float(lons[lon_idx]),
                        "curl": float(values[
                            t_idx,
                            lat_idx,
                            lon_idx,
                        ]),
                        "abs_curl": float(
                            abs(
                                values[
                                    t_idx,
                                    lat_idx,
                                    lon_idx,
                                ]
                            )
                        ),
                    }
                )

            large_df = pd.DataFrame(large_rows)

            print(
                large_df.sort_values(
                    "abs_curl",
                    ascending=False,
                ).head(TOP_N).to_string(
                    index=False,
                    formatters={
                        "curl": "{:.8e}".format,
                        "abs_curl": "{:.8e}".format,
                    },
                )
            )

        # ---------------------------------------------------------------------
        # DOMAIN EDGE CHECK
        # ---------------------------------------------------------------------

        print_header("6. DOMAIN BOUNDARY CHECK")

        lat = curl["latitude"].values
        lon = curl["longitude"].values

        boundary_mask = (
            (curl["latitude"] == lat.min())
            | (curl["latitude"] == lat.max())
            | (curl["longitude"] == lon.min())
            | (curl["longitude"] == lon.max())
        )

        boundary_values = curl.where(
            boundary_mask,
            drop=True,
        ).values

        boundary_finite = boundary_values[
            np.isfinite(boundary_values)
        ]

        boundary_abs = np.abs(boundary_finite)

        print(
            f"Finite boundary curl values: "
            f"{boundary_finite.size:,}"
        )

        if boundary_finite.size > 0:

            print(
                f"Boundary |curl| P99: "
                f"{np.percentile(boundary_abs, 99):.8e}"
            )

            print(
                f"Boundary |curl| Max: "
                f"{boundary_abs.max():.8e}"
            )

        # ---------------------------------------------------------------------
        # SST OCEAN MASK
        # ---------------------------------------------------------------------

        print_header("7. EXTREMES VS SST OCEAN MASK")

        sst_ocean = sst.notnull().any(dim="time")

        curl_valid = np.isfinite(curl)

        # Any finite curl outside the permanent SST ocean mask.
        contamination = (
            curl_valid
            & ~sst_ocean
        )

        contamination_count = int(
            contamination.sum().item()
        )

        print(
            "Finite curl values inside permanent SST-masked cells:"
        )
        print(f"  {contamination_count:,}")

        if contamination_count == 0:
            print("PASS — no curl contamination over masked cells.")
        else:
            print("FAIL — curl exists inside masked cells.")

        # ---------------------------------------------------------------------
        # EXTREMES NEAR COAST / MASK EDGE
        # ---------------------------------------------------------------------

        print_header("8. EXTREMES NEAR MASK EDGE")

        ocean = sst_ocean.values

        # A valid ocean cell is considered an edge cell if any
        # immediate 4-neighbor is land/masked.
        edge = np.zeros_like(ocean, dtype=bool)

        edge[1:, :] |= ocean[1:, :] & ~ocean[:-1, :]
        edge[:-1, :] |= ocean[:-1, :] & ~ocean[1:, :]
        edge[:, 1:] |= ocean[:, 1:] & ~ocean[:, :-1]
        edge[:, :-1] |= ocean[:, :-1] & ~ocean[:, 1:]

        curl_edge_mask = xr.DataArray(
            edge,
            coords={
                "latitude": curl["latitude"],
                "longitude": curl["longitude"],
            },
            dims=("latitude", "longitude"),
        )

        curl_edge = curl.where(curl_edge_mask)

        edge_values = curl_edge.values
        edge_finite = edge_values[
            np.isfinite(edge_values)
        ]

        if edge_finite.size > 0:

            edge_abs = np.abs(edge_finite)

            print(
                f"Finite curl values on SST mask edge: "
                f"{edge_finite.size:,}"
            )

            print(
                f"Edge |curl| P99 : "
                f"{np.percentile(edge_abs, 99):.8e}"
            )

            print(
                f"Edge |curl| Max : "
                f"{edge_abs.max():.8e}"
            )

        else:
            print("No finite curl values on mask edge.")

        # ---------------------------------------------------------------------
        # INTERIOR CURL
        # ---------------------------------------------------------------------

        print_header("9. INTERIOR VS EDGE")

        interior_mask = sst_ocean & ~curl_edge_mask

        curl_interior = curl.where(interior_mask)

        interior_values = curl_interior.values
        interior_finite = interior_values[
            np.isfinite(interior_values)
        ]

        if interior_finite.size > 0:

            interior_abs = np.abs(interior_finite)

            print(
                f"Interior finite values: "
                f"{interior_finite.size:,}"
            )

            print(
                f"Interior |curl| P99: "
                f"{np.percentile(interior_abs, 99):.8e}"
            )

            print(
                f"Interior |curl| P99.9: "
                f"{np.percentile(interior_abs, 99.9):.8e}"
            )

            print(
                f"Interior |curl| Max: "
                f"{interior_abs.max():.8e}"
            )

        # ---------------------------------------------------------------------
        # FINAL INTERPRETATION
        # ---------------------------------------------------------------------

        print_header("10. INTERPRETATION")

        global_max = abs_values.max()

        print(
            f"Global max |curl|       : {global_max:.8e}"
        )

        if edge_finite.size > 0:
            edge_max = edge_abs.max()
            print(
                f"Mask-edge max |curl|    : {edge_max:.8e}"
            )
        else:
            edge_max = 0.0

        if interior_finite.size > 0:
            interior_max = interior_abs.max()
            print(
                f"Interior max |curl|     : "
                f"{interior_max:.8e}"
            )
        else:
            interior_max = 0.0

        print()

        if contamination_count > 0:

            print(
                "FAIL: curl contamination exists over masked cells."
            )

        elif global_max > 10 * np.percentile(abs_values, 99.9):

            print(
                "REVIEW: extreme outlier is substantially above "
                "the normal curl distribution."
            )

            if edge_max >= interior_max:

                print(
                    "The strongest values are concentrated near "
                    "the SST mask edge."
                )

            else:

                print(
                    "The strongest values are not concentrated "
                    "on the SST mask edge."
                )

        else:

            print(
                "No major numerical outlier pattern detected."
            )

        print()
        print("Inspection complete.")


if __name__ == "__main__":
    main()