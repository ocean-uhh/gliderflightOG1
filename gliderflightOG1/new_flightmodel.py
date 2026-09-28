from dataclasses import dataclass

import numpy as np
import xarray as xr
import gsw
from scipy.optimize import minimize

from gliderflightOG1.utilities import construct_2dgrid, regular_grid

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FlightParameters:
    """
    Configuration parameters that are assumed constant
    over a deployment or dive.
    """

    # flight model coefficients
    hd_a: float = 0.004  #
    hd_b: float = 0.01
    hd_c: float = 5.7e-5
    xl: float = 1.8
    mass: float = 52.0  # kg
    volmax: float = 51000  # cc

    # reference density
    rho0: float = 1027.5  # kg/m^3
    gravity = 9.81  # m/s^2

    # buoyancy model
    vbd_bias: float = 0.0
    vbd_min_cnts: float = 0.0
    vbd_cnts_per_cc: float = -4.077

    # glider volume model
    temp_ref: float = 15.0
    therm_exp: float = 7.05e-5  # thermal expansion 7e-5
    abs_compress: float = 4.1e-6  # compressibility 4.4e-6

    # solver
    tol: float = 1e-3
    max_iter: int = 15


# Global variable to track optimization progress
_optimization_progress = []


class SteadyFlightModel:
    """
    Steady-state glider flight model.

    Paramters:
    ----------
    params: FlightParameters
        Configuration parameters for the flight model.
    Input:
    ------
    ds: xarray.Dataset
        Input dataset. Must contain the following variables:
        - PSAL: Practical Salinity (unitless)
        - TEMP: In-situ Temperature (°C)
        - PRESS: Pressure (dbar)
        - vbd: Variable Buoyancy Device volume (cc)
        - pitch: Glider pitch angle (degrees)

    Returns:
    -------
    ds: xarray.Dataset
        Input dataset with added variables for flight speed and angle.
    """

    def __init__(self, params: FlightParameters) -> None:
        self.params = params

    def solve_model(
        self, ds: xr.Dataset, which_par: np.ndarray = np.array([1, 1, 1, 0, 0, 0])
    ) -> xr.Dataset:
        """
        Complete workflow.
        Parameters
        ----------
        ds: xarray.Dataset
        which_par: np.ndarray
            Boolean array indicating which parameters to optimize. Order should be:
            [hd_a, hd_b, vbdbias, abs_compress, therm_expan, hd_c]
        """
        global _optimization_progress
        _optimization_progress = []  # Reset progress tracker

        logger.info("🚁 Starting Glider Flight Model Parameter Optimization")
        logger.info("=" * 60)

        p = self.params
        param_names = ["hd_a", "hd_b", "vbd_bias", "abs_compress", "therm_exp", "hd_c"]

        logger.info("🔍 Initial Parameter Values:")
        for name in param_names:
            logger.info(f"  {name}: {getattr(p, name)}")

        logger.info("\nThe following parameters will be optimized:")
        for i, name in enumerate(param_names):
            if which_par[i] == 1:
                logger.info(f"  {name}")

        # Extract only the initial values where which_par == 1
        all_initials = np.array(
            [p.hd_a, p.hd_b, p.vbd_bias, p.abs_compress, p.therm_exp, p.hd_c]
        )
        x0_to_optimize = all_initials[which_par == 1]

        # Local state tracker for iterations
        tracker = {"iteration_count": 0}

        # Define the objective function that scipy.optimize.minimize will iteratively call
        def objective(x):
            # 1. Map the optimizer's active array 'x' back to our full parameter set
            x_idx = 0
            for idx, name in enumerate(param_names):
                if which_par[idx] == 1:
                    setattr(self.params, name, x[x_idx])
                    x_idx += 1

            # 2. Run the misfit evaluation using the newly assigned parameters
            cost = self.misfit(ds)
            # 3. Track and print specific iterations
            tracker["iteration_count"] += 1
            current_iter = tracker["iteration_count"]

            # Condition: First 5 iterations OR every 10th iteration thereafter (e.g., 10, 20, 30...)
            if current_iter <= 5 or current_iter % 10 == 0:
                # Build a running counter that mirrors the order params were packed into x
                active_names = [
                    name for i, name in enumerate(param_names) if which_par[i] == 1
                ]
                active_params = ", ".join(
                    [f"{name}: {val:.4e}" for name, val in zip(active_names, x)]
                )
                logger.info(
                    f"🔄 Iteration {current_iter:03d} | Cost: {cost:.6f} | Params -> [{active_params}]"
                )
            return cost

        # Run optimization
        result = minimize(
            objective,  # Pass the dynamic wrapper function here
            x0=x0_to_optimize,
            method="Nelder-Mead",
            options={"maxfev": 250, "xatol": 0.1, "fatol": 0.01, "disp": True},
        )

        # Apply final optimal values back to parameters permanently
        x_idx = 0
        for idx, name in enumerate(param_names):
            if which_par[idx] == 1:
                setattr(self.params, name, result.x[x_idx])
                x_idx += 1

        logger.info("\n✅ Optimization Completed!")
        logger.info("Optimal Parameters:")
        for name in param_names:
            logger.info(f"  {name}: {getattr(self.params, name)}")
        logger.info(f"Final Cost: {result.fun}")

        return ds

    def misfit(self, ds):
        rho = self.compute_density(ds)
        vol = self.compute_volume(ds)

        updn = xr.where(ds.PROFILE_NUMBER % 2 == 1, -1, 1)  # 1 for up, -1 for down
        dzdt = ds.W_MEAS.values

        F_B = self.compute_buoyancy_force(
            density=rho,
            vol=vol,
        )

        umag, thdeg = self.solve_flight(
            F_B=F_B,
            pitch=ds.PITCH.values,
            updn=updn,
        )

        w_model = umag * np.sin(np.deg2rad(thdeg))
        w_water = dzdt - w_model

        wrms = self.cost_function(ds, updn, w_water)
        return wrms

    def cost_function(self, ds, updn, w_water):
        """
        Cost function to minimize: binned RMS of vertical velocity misfit between model and observations.
        """
        iup = np.where(updn > 0)[0]
        idn = np.where(updn < 0)[0]

        delta_z = 10
        delta_pn = ds.PROFILE_NUMBER.max()

        ### create two grids for up and down profile to
        ### Maybe a problem if profiles do not start and end at the same depth?
        z_grid = regular_grid(ds.DEPTH.values, delta_z)
        climb_grid, _, _ = construct_2dgrid(
            ds.DEPTH.values[iup],
            ds.PROFILE_NUMBER.values[iup],
            w_water[iup],
            z_grid,
            delta_pn,
            agg="mean",
        )
        dive_grid, _, _ = construct_2dgrid(
            ds.DEPTH.values[idn],
            ds.PROFILE_NUMBER.values[idn],
            w_water[idn],
            z_grid,
            delta_pn,
            agg="mean",
        )

        w_climb = climb_grid.flatten()
        w_dive = dive_grid.flatten()

        wrms = np.nanmean(w_climb**2 + w_dive**2)

        return wrms

    def solve_flight(
        self,
        F_B: np.ndarray,
        pitch: np.ndarray,
        updn: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Iterative q-solver.
        """

        p = self.params

        th = (np.pi / 4) * updn
        # Initial guess for dynamic pressure q based on free fall assumption
        q = (np.abs(F_B) / (p.xl**2 * p.hd_b)) ** (4 / 3)

        # Initialize arrays to store previous values and intermediate calculations
        q_old = np.zeros_like(q)
        alpha = np.zeros_like(q)
        thdeg = np.zeros_like(q)
        param = np.ones_like(q)

        # Define a boolean mask to identify valid indices for the iterative solver
        valid = (F_B != 0) & (np.sign(F_B) * np.sign(pitch) > 0)

        iteration = 0
        # Iterate until convergence or maximum iterations reached
        while (
            np.any(np.abs((q[valid] - q_old[valid]) / q[valid]) > p.tol)
            and iteration <= p.max_iter
        ):

            q_old = q.copy()

            param_inv = p.hd_a**2 * np.tan(th) ** 2 * q**0.25 / (4 * p.hd_b * p.hd_c)

            valid = (param_inv > 1) & (np.sign(F_B) * np.sign(pitch) > 0)

            param[valid] = (
                4
                * p.hd_b
                * p.hd_c
                / (p.hd_a**2 * np.tan(th[valid]) ** 2 * q[valid] ** 0.25)
            )

            q[valid] = (
                F_B[valid]
                * np.sin(th[valid])
                / (2 * p.xl**2 * p.hd_b * q[valid] ** (-0.25))
            ) * (1 + np.sqrt(1 - param[valid]))

            q = np.maximum(q, 1e-10)

            alpha[valid] = (p.hd_a * np.tan(th[valid]) / (2 * p.hd_c)) * (
                1 - np.sqrt(1 - param[valid])
            )

            if valid.any():
                thdeg[valid] = pitch[valid] + alpha[valid]
            else:
                thdeg[valid] = np.nan

            # Identify stall conditions where the model is not valid
            stall = (param_inv <= 1) | (np.sign(F_B) * np.sign(pitch) < 0)

            q[stall] = 0.0
            thdeg[stall] = 0.0

            th[valid] = np.deg2rad(thdeg[valid])

            iteration += 1

            # Convert dynamic pressure to flight speed (cm/s)
            umag = 100 * np.sqrt(2 * q / p.rho0)

        return umag, thdeg

    def compute_flight_speed(self, ds):
        """Compute flight speed from density, volume, and other parameters.

        Parameters
        ----------
        ds : xarray.Dataset

        Returns
        -------
        speed : array-like
            Flight speed (cm/s).
        """
        rho = self.compute_density(ds)
        vol = self.compute_volume(ds)

        updn = xr.where(ds.PROFILE_NUMBER % 2 == 1, -1, 1)  # 1 for up, -1 for down

        F_B = self.compute_buoyancy_force(
            density=rho,
            vol=vol,
        )

        umag, theta = self.solve_flight(
            F_B=F_B,
            pitch=ds.PITCH.values,
            updn=updn,
        )

        return umag, theta

    def compute_density(self, ds):
        """Compute in-situ density from salinity, temperature, and pressure using GSW.

        Parameters
        ----------
        ds : xarray.Dataset

        Returns
        -------
        density : array-like
            In-situ density (kg/m³).
        """
        salin = ds.PSAL.values
        temp = ds.TEMP.values
        press = ds.PRES.values
        lon = ds.LONGITUDE.values
        lat = ds.LATITUDE.values

        # Step 1: Convert Practical Salinity to Absolute Salinity
        SA = gsw.SA_from_SP(salin, press, lon, lat)

        # Step 2: Convert in-situ Temperature to Conservative Temperature
        CT = gsw.CT_from_t(SA, temp, press)

        # Step 3: Compute in-situ density
        density = gsw.rho(SA, CT, press)

        return density

    def compute_volume(self, ds):
        """Compute glider volume from VBD and other parameters.

        Parameters
        ----------
        ds : xarray.Dataset

        Returns
        -------
        volume : array-like
            Glider volume (cc).
        """
        vbd = ds.VBD.values
        c_vbd = ds.C_VBD.values
        press = ds.PRES.values
        temp = ds.TEMP.values
        p = self.params

        vol1 = vbd + p.volmax + (c_vbd - p.vbd_min_cnts) / p.vbd_cnts_per_cc
        compr_factor = np.exp(
            -p.abs_compress * press + p.therm_exp * (temp - p.temp_ref)
        )

        # vbdc = vbd - p.vbd_bias
        vol = (vol1 - p.vbd_bias) * compr_factor
        return vol

    def compute_buoyancy_force(
        self,
        density: np.ndarray,
        vol: np.ndarray,
    ) -> np.ndarray:
        """
        Compute buoyancy force B from density, volume, and other parameters.
        """

        p = self.params
        cc_to_m3 = 1e-6  # conversion factor from cc to m³

        F_B = p.gravity * (-p.mass + density * vol * cc_to_m3)

        return F_B

    def compute_lift_force(
        self,
        q: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """
        Compute lift force L from density, speed, and other parameters.
        """

        p = self.params

        F_L = q * p.xl**2 * p.hd_a * np.sin(alpha)

        return F_L

    def compute_drag_force(
        self,
        q: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """
        Compute drag force D from density, speed, and other parameters.
        """

        p = self.params

        F_D = q * p.xl**2 * (p.hd_b * q**-0.25 + p.hd_c * alpha**2)

        return F_D

    def compute_alpha(
        self,
        q: np.ndarray,
        alpha: np.ndarray,
        pitch: np.ndarray,
        updn: np.ndarray,
    ) -> np.ndarray:
        """
        Compute alpha from density, speed, and other parameters.
        """

        p = self.params

        alpha = (
            updn
            * (p.hd_b * q**-0.25 + p.hd_c * alpha**2)
            / (p.hd_a * np.tan(np.deg2rad(pitch - alpha)))
        )

        return alpha
