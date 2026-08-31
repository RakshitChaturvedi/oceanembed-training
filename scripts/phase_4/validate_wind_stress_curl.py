from pathlib import Path

import numpy as np
import xarray as xr


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WIND_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase2"
    / "wind.nc"
)

SST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase2"
    / "sst.nc"
)

CURL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase4"
    / "wind_stress_curl.nc"
)


# ============================================================================
# HELPERS
# ============================================================================

def check(condition: bool, message: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"{message:<60}: {status}")
    return condition


def print_stats(name: str, da: xr.DataArray) -> None:
    values = da.values
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        print(f"{name}: no finite values")
        return

    print(f"{name}")
    print(f"  Min       : {np.min(finite):.8e}")
    print(f"  Max       : {np.max(finite):.8e}")
    print(f"  Mean      : {np.mean(finite):.8e}")
    print(f"  Median    : {np.median(finite):.8e}")
    print(f"  Std       : {np.std(finite):.8e}")
    print(f"  P01       : {np.percentile(finite, 1):.8e}")
    print(f"  P05       : {np.percentile(finite, 5):.8e}")
    print(f"  P95       : {np.percentile(finite, 95):.8e}")
    print(f"  P99       : {np.percentile(finite, 99):.8e}")
    print(f"  P999      : {np.percentile(finite, 99.9):.8e}")
    print(f"  Finite    : {finite.size:,}")


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 90)
    print("OCEANEMBED PHASE 4 — WIND STRESS CURL VALIDATION")
    print("=" * 90)

    print()
    print(f"WIND : {WIND_PATH}")
    print(f"SST  : {SST_PATH}")
    print(f"CURL : {CURL_PATH}")
    print()

    for path in (WIND_PATH, SST_PATH, CURL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    wind_ds = xr.open_dataset(WIND_PATH)
    sst_ds = xr.open_dataset(SST_PATH)
    curl_ds = xr.open_dataset(CURL_PATH)

    try:

        failures = []

        # ====================================================================
        # 1. DATASET STRUCTURE
        # ====================================================================

        print("=" * 90)
        print("1. DATASET STRUCTURE")
        print("=" * 90)

        expected_vars = {
            "wind_stress_curl",
            "wind_stress_x",
            "wind_stress_y",
        }

        actual_vars = set(curl_ds.data_vars)

        print()
        print("Expected variables:")
        print(f"  {sorted(expected_vars)}")

        print("Actual variables:")
        print(f"  {sorted(actual_vars)}")

        ok = expected_vars.issubset(actual_vars)
        if not check(ok, "Expected variables present"):
            failures.append("Missing expected variables")

        # ====================================================================
        # 2. GRID ALIGNMENT
        # ====================================================================

        print()
        print("=" * 90)
        print("2. GRID ALIGNMENT")
        print("=" * 90)

        for coord in ("time", "latitude", "longitude"):

            ok = (
                coord in curl_ds.coords
                and coord in wind_ds.coords
                and coord in sst_ds.coords
                and curl_ds[coord].equals(wind_ds[coord])
                and curl_ds[coord].equals(sst_ds[coord])
            )

            if not check(ok, coord):
                failures.append(f"{coord} grid mismatch")

        # ====================================================================
        # 3. DIMENSIONS
        # ====================================================================

        print()
        print("=" * 90)
        print("3. DIMENSIONS")
        print("=" * 90)

        expected_dims = {
            "time": 1419,
            "latitude": 101,
            "longitude": 241,
        }

        for dim, expected_size in expected_dims.items():

            actual_size = curl_ds.sizes.get(dim)

            ok = actual_size == expected_size

            if not check(
                ok,
                f"{dim}: expected {expected_size}, got {actual_size}",
            ):
                failures.append(f"Invalid {dim} dimension")

        # =============================================================================
        # 4. OCEAN MASK VALIDATION
        # =============================================================================

        print("=" * 80)
        print("4. OCEAN MASK VALIDATION")
        print("=" * 80)

        sst_data = sst_ds["sst"]
        curl_data = curl_ds["wind_stress_curl"]

        # Permanent SST ocean mask:
        # A cell is considered ocean if it has at least one valid SST observation.
        sst_ocean_mask = sst_data.notnull().any(dim="time")

        # Curl has a time dimension, so determine whether a cell has at least
        # one finite curl observation.
        curl_valid_mask = np.isfinite(curl_data).any(dim="time")

        expected_ocean = int(sst_ocean_mask.sum().item())
        expected_masked = int((~sst_ocean_mask).sum().item())

        curl_valid = int(curl_valid_mask.sum().item())
        curl_masked = int((~curl_valid_mask).sum().item())

        # Any curl value outside the SST-defined ocean is contamination.
        outside_ocean = curl_valid_mask & ~sst_ocean_mask

        outside_ocean_count = int(outside_ocean.sum().item())

        # Ocean cells which have no valid curl.
        ocean_without_curl = sst_ocean_mask & ~curl_valid_mask

        ocean_without_curl_count = int(ocean_without_curl.sum().item())

        print()
        print(f"Expected ocean cells : {expected_ocean:,}")
        print(f"Expected masked cells: {expected_masked:,}")
        print(f"Curl valid cells     : {curl_valid:,}")
        print(f"Curl masked cells    : {curl_masked:,}")
        print()

        print(
            f"Curl values outside SST ocean mask: "
            f"{outside_ocean_count:,}"
        )

        if outside_ocean_count == 0:
            print(
                "Curl valid region is fully contained within SST ocean mask "
                " : PASS"
            )
        else:
            print(
                "Curl valid region is fully contained within SST ocean mask "
                " : FAIL"
            )

        print()
        print(
            f"SST ocean cells without valid curl: "
            f"{ocean_without_curl_count:,}"
        )

        if ocean_without_curl_count == 0:
            print(
                "All SST ocean cells have curl "
                " : PASS"
            )
        else:
            print(
                "Some SST ocean cells lack curl because of masked neighbors "
                " : EXPECTED"
            )

        # Specifically identify the lost ocean cells.
        curl_edge_loss = ocean_without_curl

        curl_edge_loss_count = int(curl_edge_loss.sum().item())

        print()
        print(
            f"Ocean cells excluded from curl derivative: "
            f"{curl_edge_loss_count:,}"
        )

        if curl_edge_loss_count == 886:
            print(
                "Expected coastal/edge derivative loss observed "
                " : PASS"
            )
        else:
            print(
                "Derivative-loss count differs from previous diagnostic "
                " : REVIEW"
            )

        # ====================================================================
        # 5. RAW NaN / INF CHECK
        # ====================================================================

        print()
        print("=" * 90)
        print("5. NaN / INF CHECK")
        print("=" * 90)

        for name in (
            "wind_stress_x",
            "wind_stress_y",
            "wind_stress_curl",
        ):

            da = curl_ds[name]

            nan_count = int(da.isnull().sum().values)
            inf_count = int(
                np.isinf(da.values).sum()
            )

            print()
            print(name)
            print(f"  NaNs: {nan_count:,}")
            print(f"  Infs: {inf_count:,}")

            ok = inf_count == 0

            if not check(ok, f"{name} has no Infs"):
                failures.append(f"{name} contains Inf")

        # ====================================================================
        # 6. MASKED CELLS MUST REMAIN NaN
        # ====================================================================

        print()
        print("=" * 90)
        print("6. MASKED CELL INTEGRITY")
        print("=" * 90)

        masked_curl = curl_data.where(~sst_ocean_mask)

        masked_finite = np.isfinite(
            masked_curl.values
        ).sum()

        print(
            f"Finite curl values inside masked cells: "
            f"{masked_finite:,}"
        )

        ok = masked_finite == 0

        if not check(
            ok,
            "No curl values exist inside masked cells",
        ):
            failures.append("Masked cells contain curl values")

        # ====================================================================
        # 7. WIND STRESS STATISTICS
        # ====================================================================

        print()
        print("=" * 90)
        print("7. WIND STRESS STATISTICS")
        print("=" * 90)

        print()

        print_stats(
            "Zonal wind stress [N m-2]",
            curl_ds["wind_stress_x"],
        )

        print()

        print_stats(
            "Meridional wind stress [N m-2]",
            curl_ds["wind_stress_y"],
        )

        # Check for absurd values.
        #
        # Surface wind stresses above several N/m² would be extremely
        # unusual for this dataset. We use a deliberately generous
        # threshold so this is a sanity check rather than a scientific
        # cutoff.
        tau_x_abs_max = float(
            np.nanmax(np.abs(curl_ds["wind_stress_x"].values))
        )

        tau_y_abs_max = float(
            np.nanmax(np.abs(curl_ds["wind_stress_y"].values))
        )

        print()
        print(f"Max |tau_x|: {tau_x_abs_max:.6f} N/m²")
        print(f"Max |tau_y|: {tau_y_abs_max:.6f} N/m²")

        stress_ok = (
            tau_x_abs_max < 10.0
            and tau_y_abs_max < 10.0
        )

        if not check(
            stress_ok,
            "Wind stress magnitudes are within broad sanity limits",
        ):
            failures.append("Wind stress magnitude suspicious")

        # ====================================================================
        # 8. CURL STATISTICS
        # ====================================================================

        print()
        print("=" * 90)
        print("8. WIND STRESS CURL STATISTICS")
        print("=" * 90)

        print()

        print_stats(
            "Wind stress curl [N m-3]",
            curl_data,
        )

        curl_abs_max = float(
            np.nanmax(np.abs(curl_data.values))
        )

        curl_abs_p999 = float(
            np.nanpercentile(
                np.abs(curl_data.values),
                99.9,
            )
        )

        print()
        print(f"Max |curl| : {curl_abs_max:.8e} N/m³")
        print(f"P99.9 |curl|: {curl_abs_p999:.8e} N/m³")

        # A very broad numerical sanity threshold.
        #
        # This is intentionally NOT being used as a scientific truth
        # threshold. It is only intended to catch derivative explosions
        # caused by unit mistakes or coastline discontinuities.
        curl_ok = curl_abs_max < 1e-3

        if not check(
            curl_ok,
            "Curl magnitude passes broad numerical sanity check",
        ):
            failures.append(
                "Curl contains potentially extreme values"
            )

        # ====================================================================
        # 9. COASTAL / MASK-EDGE SPIKE CHECK
        # ====================================================================

        print()
        print("=" * 90)
        print("9. MASK-EDGE / COASTAL SPIKE CHECK")
        print("=" * 90)

        # A cell is considered an edge cell when it is ocean but has
        # at least one neighboring masked cell.
        #
        # This lets us compare curl behavior near coastlines against
        # the interior ocean.

        mask = sst_ocean_mask.values

        edge_mask = np.zeros_like(mask, dtype=bool)

        # North/south neighbors
        edge_mask[1:, :] |= mask[1:, :] & ~mask[:-1, :]
        edge_mask[:-1, :] |= mask[:-1, :] & ~mask[1:, :]

        # East/west neighbors
        edge_mask[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
        edge_mask[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]

        edge_ocean = edge_mask
        interior_ocean = mask & ~edge_mask

        curl_abs = np.abs(curl_data.values)

        edge_values = curl_abs[:, edge_ocean]
        interior_values = curl_abs[:, interior_ocean]

        edge_values = edge_values[
            np.isfinite(edge_values)
        ]

        interior_values = interior_values[
            np.isfinite(interior_values)
        ]

        print()
        print(f"Ocean edge cells   : {edge_ocean.sum():,}")
        print(f"Interior ocean cells: {interior_ocean.sum():,}")

        if edge_values.size > 0:
            edge_p99 = np.percentile(
                edge_values,
                99,
            )

            edge_p999 = np.percentile(
                edge_values,
                99.9,
            )

            edge_max = np.max(edge_values)

            print()
            print("Coastal-edge |curl|:")
            print(f"  P99   : {edge_p99:.8e}")
            print(f"  P99.9 : {edge_p999:.8e}")
            print(f"  Max   : {edge_max:.8e}")

        if interior_values.size > 0:
            interior_p99 = np.percentile(
                interior_values,
                99,
            )

            interior_p999 = np.percentile(
                interior_values,
                99.9,
            )

            interior_max = np.max(interior_values)

            print()
            print("Interior |curl|:")
            print(f"  P99   : {interior_p99:.8e}")
            print(f"  P99.9 : {interior_p999:.8e}")
            print(f"  Max   : {interior_max:.8e}")

        if edge_values.size > 0 and interior_values.size > 0:

            edge_p99 = np.percentile(
                edge_values,
                99,
            )

            interior_p99 = np.percentile(
                interior_values,
                99,
            )

            ratio = (
                edge_p99 / interior_p99
                if interior_p99 > 0
                else np.inf
            )

            print()
            print(
                f"Coastal/interior P99 ratio: {ratio:.3f}"
            )

            # We don't fail merely because coastal curl is larger.
            # Coastlines legitimately produce larger gradients.
            #
            # This is only a warning threshold for obvious numerical
            # discontinuities.
            if ratio > 20:
                print(
                    "WARNING: Coastal curl is substantially larger "
                    "than interior curl."
                )
                print(
                    "Inspect spatial distribution before proceeding."
                )

        # ====================================================================
        # 10. DOMAIN EDGE CHECK
        # ====================================================================

        print()
        print("=" * 90)
        print("10. DOMAIN BOUNDARY CHECK")
        print("=" * 90)

        boundary_mask = np.zeros_like(mask, dtype=bool)

        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True

        boundary_ocean = boundary_mask & mask

        boundary_values = curl_abs[:, boundary_ocean]
        boundary_values = boundary_values[
            np.isfinite(boundary_values)
        ]

        if boundary_values.size > 0:

            boundary_max = np.max(boundary_values)
            boundary_p99 = np.percentile(
                boundary_values,
                99,
            )

            print()
            print(
                f"Ocean boundary cells: "
                f"{boundary_ocean.sum():,}"
            )
            print(
                f"Boundary P99 |curl|: "
                f"{boundary_p99:.8e}"
            )
            print(
                f"Boundary max |curl|: "
                f"{boundary_max:.8e}"
            )

            print()
            print(
                "Boundary values use one-sided numerical "
                "differences through xarray.differentiate()."
            )

        # ====================================================================
        # 11. TIME COVERAGE
        # ====================================================================

        print()
        print("=" * 90)
        print("11. TEMPORAL COVERAGE")
        print("=" * 90)

        time_count = curl_ds.sizes["time"]

        print()
        print(f"Time count: {time_count}")
        print(
            f"Start: {curl_ds.time.values[0]}"
        )
        print(
            f"End  : {curl_ds.time.values[-1]}"
        )

        time_diffs = np.diff(
            curl_ds.time.values
        ).astype("timedelta64[D]").astype(int)

        ok = (
            len(time_diffs) > 0
            and np.all(time_diffs == 1)
        )

        if not check(
            ok,
            "Daily time continuity",
        ):
            failures.append("Time continuity failure")

        # ====================================================================
        # 12. ATTRIBUTE CHECK
        # ====================================================================

        print()
        print("=" * 90)
        print("12. METADATA")
        print("=" * 90)

        required_attrs = [
            "oceanembed_phase",
            "oceanembed_feature",
            "oceanembed_formula",
            "oceanembed_ocean_mask",
            "oceanembed_mask_before_derivative",
            "oceanembed_air_density_kg_m3",
            "oceanembed_drag_coefficient",
            "oceanembed_earth_radius_m",
        ]

        for attr in required_attrs:

            ok = attr in curl_ds.attrs

            if not check(
                ok,
                attr,
            ):
                failures.append(
                    f"Missing metadata: {attr}"
                )

        if "oceanembed_mask_before_derivative" in curl_ds.attrs:
            value = curl_ds.attrs[
                "oceanembed_mask_before_derivative"
            ]

            print()
            print(
                "Mask-before-derivative metadata: "
                f"{value}"
            )

        # ====================================================================
        # 13. FORMULA CHECK
        # ====================================================================

        print()
        print("=" * 90)
        print("13. FORMULA")
        print("=" * 90)

        formula = curl_ds.attrs.get(
            "oceanembed_formula",
            "",
        )

        print()
        print(f"Recorded formula: {formula}")

        expected_formula = (
            "curl_tau = d(tau_y)/dx - d(tau_x)/dy"
        )

        ok = formula == expected_formula

        if not check(
            ok,
            "Formula metadata",
        ):
            failures.append("Formula metadata mismatch")

        # ====================================================================
        # 14. FINAL RESULT
        # ====================================================================

        print()
        print("=" * 90)
        print("FINAL VALIDATION RESULT")
        print("=" * 90)

        if failures:

            print()
            print("FAIL")

            print()
            print("Issues found:")

            for failure in failures:
                print(f"  - {failure}")

            print()
            print(
                "Do NOT proceed to the next phase until these "
                "issues are investigated."
            )

            return 1

        print()
        print("PASS")
        print()
        print(
            "Phase 4 wind-stress curl passed structural and "
            "numerical validation."
        )
        print()
        print(
            "Recommended next step: inspect the spatial curl "
            "distribution before proceeding to model-input construction."
        )

        return 0

    finally:
        wind_ds.close()
        sst_ds.close()
        curl_ds.close()


if __name__ == "__main__":
    raise SystemExit(main())