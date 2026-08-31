from __future__ import annotations

import numpy as np
import xarray as xr


# Earth radius [m]
EARTH_RADIUS = 6_371_000.0

# Air density [kg/m^3]
AIR_DENSITY = 1.225

# Standard bulk drag coefficient.
#
# For this lightweight implementation we use a constant Cd.
# This is adequate for a derived feature; there is no need to
# introduce a complicated atmospheric drag parameterization here.
DRAG_COEFFICIENT = 1.3e-3


def compute_drag_coefficient(wind_speed: xr.DataArray) -> xr.DataArray:
    """
    Return the surface wind drag coefficient.

    A constant bulk coefficient is used for the Phase 4 feature.
    """
    return xr.full_like(wind_speed, DRAG_COEFFICIENT)


def compute_wind_stress(
    wind_u: xr.DataArray,
    wind_v: xr.DataArray,
    wind_speed: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute wind stress components:

        tau_x = rho_air * Cd * |U| * U
        tau_y = rho_air * Cd * |U| * V

    Units: N/m^2
    """

    cd = compute_drag_coefficient(wind_speed)

    tau_x = AIR_DENSITY * cd * wind_speed * wind_u
    tau_y = AIR_DENSITY * cd * wind_speed * wind_v

    tau_x.name = "wind_stress_x"
    tau_y.name = "wind_stress_y"

    tau_x.attrs.update(
        {
            "long_name": "zonal wind stress",
            "units": "N m-2",
            "air_density": AIR_DENSITY,
            "drag_coefficient": DRAG_COEFFICIENT,
        }
    )

    tau_y.attrs.update(
        {
            "long_name": "meridional wind stress",
            "units": "N m-2",
            "air_density": AIR_DENSITY,
            "drag_coefficient": DRAG_COEFFICIENT,
        }
    )

    return tau_x, tau_y


def compute_wind_stress_curl(
    tau_x: xr.DataArray,
    tau_y: xr.DataArray,
) -> xr.DataArray:
    """
    Compute wind stress curl on a latitude/longitude grid.

        curl_tau = d(tau_y)/dx - d(tau_x)/dy

    Because the data are on a spherical Earth:

        dx = R cos(latitude) d(lambda)
        dy = R d(phi)

    xarray.differentiate() operates with respect to the coordinate
    values, so longitude/latitude derivatives are converted to
    physical distances afterward.

    Interior points use centered differences.
    Boundary points use one-sided differences through xarray's
    numerical differentiation.
    """

    lat_rad = np.deg2rad(tau_x["latitude"])

    # Derivative with respect to longitude [N/m² per degree]
    d_tau_y_dlon = tau_y.differentiate("longitude")

    # Derivative with respect to latitude [N/m² per degree]
    d_tau_x_dlat = tau_x.differentiate("latitude")

    # Convert degree derivatives to radian derivatives.
    #
    # d/d(rad) = d/d(degree) * 180/pi
    degree_to_radian = 180.0 / np.pi

    d_tau_y_dlambda = d_tau_y_dlon * degree_to_radian
    d_tau_x_dphi = d_tau_x_dlat * degree_to_radian

    # Physical distances:
    #
    # dx = R cos(phi) d(lambda)
    # dy = R d(phi)
    #
    # Therefore:
    #
    # d/dx = 1/(R cos(phi)) d/dlambda
    # d/dy = 1/R d/dphi

    cos_lat = np.cos(lat_rad)

    d_tau_y_dx = d_tau_y_dlambda / (
        EARTH_RADIUS * cos_lat
    )

    d_tau_x_dy = d_tau_x_dphi / EARTH_RADIUS

    curl = d_tau_y_dx - d_tau_x_dy

    curl.name = "wind_stress_curl"

    curl.attrs.update(
        {
            "long_name": "curl of surface wind stress",
            "standard_name": "wind_stress_curl",
            "units": "N m-3",
            "formula": "d(tau_y)/dx - d(tau_x)/dy",
            "earth_radius_m": EARTH_RADIUS,
            "numerical_differentiation": "xarray.differentiate",
            "interior_difference": "centered",
            "boundary_difference": "one-sided",
        }
    )

    return curl


def process_wind_stress_curl(
    wind_path: str,
    sst_path: str,
    output_path: str,
) -> None:
    """
    Complete Phase 4 pipeline.

    1. Load wind and SST.
    2. Verify grids match.
    3. Build permanent ocean mask from SST.
    4. Mask wind before derivatives.
    5. Compute wind stress.
    6. Compute spherical wind-stress curl.
    7. Save output.
    """

    print(f"Opening wind: {wind_path}")
    print(f"Opening SST : {sst_path}")

    wind_ds = xr.open_dataset(wind_path)
    sst_ds = xr.open_dataset(sst_path)

    try:
        wind_u = wind_ds["wind_u"]
        wind_v = wind_ds["wind_v"]
        wind_speed = wind_ds["wind_speed"]
        sst = sst_ds["sst"]

        # ----------------------------------------------------------
        # Grid validation
        # ----------------------------------------------------------

        for coord in ("time", "latitude", "longitude"):
            if not wind_ds[coord].equals(sst_ds[coord]):
                raise ValueError(
                    f"Grid mismatch in coordinate: {coord}"
                )

        print("Grid alignment: PASS")

        # ----------------------------------------------------------
        # Permanent SST mask
        #
        # True = ocean cell with at least one valid SST observation
        # False = permanently missing / land-like cell
        # ----------------------------------------------------------

        ocean_mask = sst.notnull().any(dim="time")

        ocean_count = int(ocean_mask.sum().values)
        total_cells = ocean_mask.size
        land_count = total_cells - ocean_count

        print()
        print("Ocean mask:")
        print(f"  Ocean cells: {ocean_count:,}")
        print(f"  Masked cells: {land_count:,}")
        print(
            f"  Ocean fraction: "
            f"{100.0 * ocean_count / total_cells:.2f}%"
        )

        # ----------------------------------------------------------
        # CRITICAL:
        # Mask wind BEFORE calculating derivatives.
        #
        # Do not replace masked cells with zero.
        # NaNs must remain NaNs so that land does not create
        # artificial spatial gradients.
        # ----------------------------------------------------------

        wind_u_masked = wind_u.where(ocean_mask)
        wind_v_masked = wind_v.where(ocean_mask)
        wind_speed_masked = wind_speed.where(ocean_mask)

        # ----------------------------------------------------------
        # Wind stress
        # ----------------------------------------------------------

        print()
        print("Computing wind stress...")

        tau_x, tau_y = compute_wind_stress(
            wind_u_masked,
            wind_v_masked,
            wind_speed_masked,
        )

        # ----------------------------------------------------------
        # Wind stress curl
        # ----------------------------------------------------------

        print("Computing wind stress curl...")

        curl = compute_wind_stress_curl(
            tau_x,
            tau_y,
        )

        # Explicitly retain the ocean mask on the final product.
        curl = curl.where(ocean_mask)

        # ----------------------------------------------------------
        # Output dataset
        # ----------------------------------------------------------

        output_ds = xr.Dataset(
            {
                "wind_stress_curl": curl,
                "wind_stress_x": tau_x,
                "wind_stress_y": tau_y,
            }
        )

        # ----------------------------------------------------------
        # Metadata
        # ----------------------------------------------------------

        output_ds.attrs.update(
            {
                "oceanembed_phase": "phase4",
                "oceanembed_feature": "wind_stress_curl",
                "oceanembed_source_wind": wind_path,
                "oceanembed_source_sst": sst_path,
                "oceanembed_ocean_mask": ("Permanent valid-SST mask"),
                "oceanembed_mask_before_derivative": "true",
                "oceanembed_drag_coefficient": (DRAG_COEFFICIENT),
                "oceanembed_air_density_kg_m3": AIR_DENSITY,
                "oceanembed_earth_radius_m": EARTH_RADIUS,
                "oceanembed_formula": ("curl_tau = d(tau_y)/dx - d(tau_x)/dy"),
            }
        )

        # Keep coordinates encoded sensibly.
        output_ds["latitude"].attrs.update(
            {"units": "degrees_north"}
        )
        output_ds["longitude"].attrs.update(
            {"units": "degrees_east"}
        )

        print()
        print(f"Writing: {output_path}")

        output_ds.to_netcdf(
            output_path,
            mode="w",
        )

        output_ds.close()

        print()
        print("Phase 4 computation complete.")

    finally:
        wind_ds.close()
        sst_ds.close()