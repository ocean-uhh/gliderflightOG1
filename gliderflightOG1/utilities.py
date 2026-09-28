from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
from urllib.parse import urlparse

import xarray as xr
import requests
import numpy as np
import pandas as pd

from gliderflightOG1 import logger
from gliderflightOG1.logger import log_debug, log_error, log_info

log = logger.log

# OG1 Variable Requirements for Flight Model Functions
OG1_FLIGHT_MODEL_VARS = {
    "flightvec_ds": {
        "required": ["PITCH", "DEPTH", "TIME"],  # Basic flight model variables
        "optional": ["BUOYANCY", "RHO0"],  # Can be calculated if missing
        "description": "Core flight model calculation requiring pitch angle and depth",
    },
    "regress_all_vec": {
        "required": ["PITCH", "DEPTH", "TIME", "DIVENUM", "UPDN"],
        "optional": ["TEMP", "PSAL", "GLIDER_VERT_VELO_DZDT", "PRES"],
        "description": "Flight model parameter optimization requiring dive profiles",
    },
}


def _check_necessary_variables(ds: xr.Dataset, vars: list):
    """Checks that all of a list of variables are present in a dataset.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset that should be checked
    vars: list
        List of variables

    Raises
    ------
    KeyError:
        Raises an error if all vars not present in ds

    Notes
    -----
    Original Author: Callum Rollo

    """
    missing_vars = set(vars).difference(set(ds.variables))
    if missing_vars:
        msg = f"Required variables {list(missing_vars)} do not exist in the supplied dataset."
        raise KeyError(msg)


def check_og1_flight_variables(ds: xr.Dataset, function_name: str):
    """Check that a dataset contains required OG1 variables for flight model functions.

    Parameters
    ----------
    ds : xr.Dataset
        OG1 format glider dataset to validate
    function_name : str
        Name of the flight model function (key in OG1_FLIGHT_MODEL_VARS)

    Raises
    ------
    KeyError
        If required variables are missing
    ValueError
        If function_name is not recognized

    Returns
    -------
    missing_optional : list
        List of optional variables that are missing (for informational purposes)
    """
    if function_name not in OG1_FLIGHT_MODEL_VARS:
        raise ValueError(
            f"Unknown function '{function_name}'. Available: {list(OG1_FLIGHT_MODEL_VARS.keys())}"
        )

    requirements = OG1_FLIGHT_MODEL_VARS[function_name]

    # Check required variables
    _check_necessary_variables(ds, requirements["required"])

    # Check optional variables (don't raise error, just return list)
    missing_optional = set(requirements["optional"]).difference(set(ds.variables))

    if missing_optional:
        log_info(
            f"Optional variables missing for {function_name}: {list(missing_optional)}"
        )

    return list(missing_optional)


def get_default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def apply_defaults(default_source: str, default_files: List[str]) -> Callable:
    """Decorator to apply default values for 'source' and 'file_list' parameters if they are None.

    Parameters
    ----------
    default_source : str
        Default source URL or path.
    default_files : list of str
        Default list of filenames.

    Returns
    -------
    Callable
        A wrapped function with defaults applied.

    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(
            source: Optional[str] = None,
            file_list: Optional[List[str]] = None,
            *args,
            **kwargs,
        ) -> Callable:
            if source is None:
                source = default_source
            if file_list is None:
                file_list = default_files
            return func(source=source, file_list=file_list, *args, **kwargs)

        return wrapper

    return decorator


def _is_valid_url(url: str) -> bool:
    """Validate if a given string is a valid URL with supported schemes.

    Parameters
    ----------
    url : str
        The URL string to validate.

    Returns
    -------
    bool
        True if the URL is valid and uses a supported scheme ('http', 'https', 'ftp'),
        otherwise False.

    """
    try:
        result = urlparse(url)
        return all(
            [
                result.scheme in ("http", "https", "ftp"),
                result.netloc,
                result.path,  # Ensure there's a path, not necessarily its format
            ],
        )
    except Exception:
        return False


def resolve_file_path(
    file_name: str,
    source: Union[str, Path, None],
    download_url: Optional[str],
    local_data_dir: Path,
    redownload: bool = False,
) -> Path:
    """Resolve the path to a data file, using local source, cache, or downloading if necessary.

    Parameters
    ----------
    file_name : str
        The name of the file to resolve.
    source : str or Path or None
        Optional local source directory.
    download_url : str or None
        URL to download the file if needed.
    local_data_dir : Path
        Directory where downloaded files are stored.
    redownload : bool, optional
        If True, force redownload even if cached file exists.

    Returns
    -------
    Path
        Path to the resolved file.

    """
    # Use local source if provided
    if source and not _is_valid_url(source):
        source_path = Path(source)
        candidate_file = source_path / file_name
        if candidate_file.exists():
            log_info("Using local file: %s", candidate_file)
            return candidate_file
        else:
            log_error("Local file not found: %s", candidate_file)
            raise FileNotFoundError(f"Local file not found: {candidate_file}")

    # Use cached file if available and redownload is False
    cached_file = local_data_dir / file_name
    if cached_file.exists() and not redownload:
        log_info("Using cached file: %s", cached_file)
        return cached_file

    # Download if URL is provided
    if download_url:
        try:
            log_info("Downloading file from %s to %s", download_url, local_data_dir)
            return download_file(download_url, local_data_dir, redownload=redownload)
        except Exception as e:
            log_error("Failed to download %s: %s", download_url, e)
            raise FileNotFoundError(f"Failed to download {download_url}: {e}")

    # If no options succeeded
    raise FileNotFoundError(
        f"File {file_name} could not be resolved from local source, cache, or remote URL.",
    )


def download_file(url: str, dest_folder: str, redownload: bool = False) -> str:
    """Download a file from HTTP(S) or FTP to the specified destination folder.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    dest_folder : str
        Local folder to save the downloaded file.
    redownload : bool, optional
        If True, force re-download of the file even if it exists.

    Returns
    -------
    str
        The full path to the downloaded file.

    Raises
    ------
    ValueError
        If the URL scheme is unsupported.

    """
    dest_folder_path = Path(dest_folder)
    dest_folder_path.mkdir(parents=True, exist_ok=True)

    local_filename = dest_folder_path / Path(url).name
    if local_filename.exists() and not redownload:
        # File exists and redownload not requested
        return str(local_filename)

    parsed_url = urlparse(url)

    if parsed_url.scheme in ("http", "https"):
        # HTTP(S) download
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    elif parsed_url.scheme == "ftp":
        # FTP download
        with FTP(parsed_url.netloc) as ftp:
            ftp.login()  # anonymous login
            with open(local_filename, "wb") as f:
                ftp.retrbinary(f"RETR {parsed_url.path}", f.write)

    else:
        raise ValueError(f"Unsupported URL scheme in {url}")

    return str(local_filename)


def safe_update_attrs(
    ds: xr.Dataset,
    new_attrs: Dict[str, str],
    overwrite: bool = False,
    verbose: bool = True,
) -> xr.Dataset:
    """Safely update attributes of an xarray Dataset without overwriting existing keys,
    unless explicitly allowed.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset whose attributes will be updated.
    new_attrs : dict of str
        Dictionary of new attributes to add.
    overwrite : bool, optional
        If True, allow overwriting existing attributes. Defaults to False.
    verbose : bool, optional
        If True, emit a warning when skipping existing attributes. Defaults to True.

    Returns
    -------
    xr.Dataset
        The dataset with updated attributes.

    """
    for key, value in new_attrs.items():
        if key in ds.attrs:
            if not overwrite:
                if verbose:
                    log_debug(
                        f"Attribute '{key}' already exists in dataset attrs and will not be overwritten.",
                    )
                continue  # Skip assignment
        ds.attrs[key] = value

    return ds


def regular_grid(data, res):
    """
    Create bin edges for a regular grid such that multiples of `res` fall on
    bin centers. For example, with res=5 the edges are [-2.5, 2.5, 7.5, ...]

    Parameters
    ----------
    data : array-like
        Input data used to determine the grid range.
    res : float
        Grid resolution (bin width).

    Returns
    -------
    numpy.ndarray
        Bin edges for the regular grid.
    """
    d_min = np.floor(np.nanmin(data) / res) * res + res / 2
    d_max = np.ceil(np.nanmax(data) / res) * res - res / 2
    return np.arange(d_min, d_max + res + 1, res)


def construct_2dgrid(
    x,
    y,
    v,
    xi=1,
    yi=1,
    x_bin_center: bool = True,
    y_bin_center: bool = True,
    agg: str = "median",
):
    """
    Constructs a 2D gridded representation of input data based on specified resolutions. The function takes in x, y, and v data,
    and generates a grid where each cell contains the aggregated value (e.g., mean, median) of v corresponding to the x and y coordinates.
    If the input data is already binned and you want the grid coordinates to align with the original bin edges, set `x_bin_center` and `y_bin_center` to False and the
    resolution (i.e. xi and yi) to the bin size.

    Parameters
    ----------
    x : array-like
        Input data representing the x-dimension.
    y : array-like
        Input data representing the y-dimension.
    v : array-like
        Input data representing the z-dimension (values to be gridded).
    xi : int or float, optional, default=1
        Resolution for the x-dimension grid spacing.
    yi : int or float, optional, default=1
        Resolution for the y-dimension grid spacing.
    x_bin_center : bool, optional, default=True
        If True, the x-coordinate grid (`XI`) corresponds to the **center** of each x-bin.
        If False, it corresponds to the **left edge** of each bin.
        This is especially useful if the input `x` data is already binned with the same resolution as `xi`,
        and you want the grid coordinates to align with the original bin edges. (e.g. profile numbers).
    y_bin_center : bool, optional, default=True
        Same as `x_bin_center`, but for the y-coordinate grid (`YI`).
        Set to False if your `y` data is already pre-binned with the same resolution as `yi`.
    agg : str, optional, default='median'
        Aggregation method to be used for gridding. Options include 'mean', 'median', etc.

    Returns
    -------
    grid : numpy.ndarray
        Gridded representation of the z-values over the x and y space.
    XI : numpy.ndarray
        Gridded x-coordinates corresponding to the specified resolution.
    YI : numpy.ndarray
        Gridded y-coordinates corresponding to the specified resolution.

    Notes
    -----
    Original Author: Bastien Queste
    [Source Code](https://github.com/bastienqueste/gliderad2cp/blob/de0652f70f4768c228f83480fa7d1d71c00f9449/gliderad2cp/process_adcp.py#L140)

    Modified by Till Moritz: added the aggregation parameter and the option to chose either bin center or bin edge as the grid coordinates.
    """
    if np.size(xi) == 1:
        xi = np.arange(np.nanmin(x), np.nanmax(x) + xi + 1, xi)
    if np.size(yi) == 1:
        yi = np.arange(np.nanmin(y), np.nanmax(y) + yi + 1, yi)

    raw = pd.DataFrame({"x": x, "y": y, "v": v}).dropna()
    grid = np.full([len(xi) - 1, len(yi) - 1], np.nan)

    raw["xbins"], xbin_iter = pd.cut(
        raw.x, xi, retbins=True, labels=False, include_lowest=True, right=False
    )
    raw["ybins"], ybin_iter = pd.cut(
        raw.y, yi, retbins=True, labels=False, include_lowest=True, right=False
    )

    raw = raw.dropna(subset=["xbins", "ybins"])  # Remove out-of-bound rows
    _tmp = raw.groupby(["xbins", "ybins"])["v"].agg(agg)
    grid[
        _tmp.index.get_level_values(0).astype(int),
        _tmp.index.get_level_values(1).astype(int),
    ] = _tmp.values
    # Match XI and YI shape to grid using bin centers
    if x_bin_center:
        xi = xi[:-1] + np.diff(xi) / 2
    else:
        xi = xi[:-1]
    if y_bin_center:
        yi = yi[:-1] + np.diff(yi) / 2
    else:
        yi = yi[:-1]
    YI, XI = np.meshgrid(yi, xi)
    return grid, XI, YI


def interpolate_over_nans(ds, var_name, method="linear"):
    """
    Interpolates over NaN values in a specified variable of an xarray Dataset along the 'N_MEASUREMENTS' dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing the variable to interpolate.
    var_name : str
        The name of the variable in the dataset to interpolate.
    method : str, optional
        The interpolation method to use (default is 'linear'). Other methods supported by xarray can be used.

    Returns
    -------
    xarray.Dataset
        A new dataset with the specified variable interpolated over NaN values.
    """
    ds = ds.copy()  # avoid modifying original dataset
    var = ds[var_name]
    ds[var_name] = var.interpolate_na(dim = "N_MEASUREMENTS", method=method)
    return ds


def restore_masked_array(values, mask, fill_value=np.nan):
    """
    Restore an array to its original length using a boolean mask.

    Parameters
    ----------
    values : np.ndarray
        Values corresponding to mask == True.
    mask : np.ndarray
        Boolean mask of the original array.
    fill_value : float, default=np.nan
        Value to insert where mask == False.

    Returns
    -------
    full_array : np.ndarray
        Array of the same length as mask.
    """
    full_array = np.full(mask.shape, fill_value, dtype=float)
    full_array[mask] = values
    return full_array


def print_profile_diffs(ds, vars_to_print=("PITCH", "TEMP"), n=20, position="beginning", profiles = None):
    """
    Print timestamps, time differences, and selected variables for each profile.

    Parameters
    ----------
    ds : xarray.Dataset
    vars_to_print : list/tuple of str
        Variables to print (e.g., ["PITCH", "TEMP"])
    n : int
        Number of samples per profile
    position : str
        "beginning", "end", or "middle" — where to select the n points
    """
    if profiles is None:
        profiles = np.unique(ds.PROFILE_NUMBER.values)

    elif isinstance(profiles, int) or isinstance(profiles, float):
        profiles = [profiles]

    else:
        print("Warning: profiles should be an int or list of ints. Using all profiles.")
        profiles = np.unique(ds.PROFILE_NUMBER.values)

    for profile in profiles:
        print(f"\nProfile {profile}:")

        # mask for this profile
        profile_msk = ds.PROFILE_NUMBER.values == profile

        # timestamps → numeric int64 (ns)
        ts = ds.TIME.values[profile_msk].astype("datetime64[ns]").astype("int64")

        L = len(ts)

        # --- choose region ---
        if position == "beginning":
            idx = slice(0, min(n, L))

        elif position == "end":
            idx = slice(max(0, L - n), L)

        elif position == "middle":
            if n >= L:
                idx = slice(0, L)
            else:
                mid = L // 2
                half = n // 2
                start = max(0, mid - half)
                end = min(L, start + n)
                idx = slice(start, end)

        else:
            raise ValueError("position must be 'beginning', 'end', or 'middle'")

        # selected timestamps
        ts_sel = ts[idx]

        # compute Δt (ns → s)
        diff_seconds = np.diff(ts_sel) * 1e-9

        # collect selected variable data (aligned with diff → skip first)
        var_data = {}
        for var in vars_to_print:
            var_values = ds[var].values[profile_msk][idx]
            var_data[var] = var_values[1:]

        # print header
        header = f"{'t datetime[ns]':<29} | {'Δt [s]':>10}"
        for var in vars_to_print:
            header += f" | {var:>10}"
        print(header)
        print("-" * len(header))

        # rows
        for i, dt in enumerate(diff_seconds):
            t = ts_sel[i + 1].astype("datetime64[ns]")
            row = f"{str(t):<29} | {dt:10.4f}"

            for var in vars_to_print:
                val = var_data[var][i]
                try:
                    row += f" | {float(val):10.4f}"
                except Exception:
                    row += f" | {str(val):>10}"

            print(row)



def add_heights(ds, bathymetry = None):
    """
    Add height above apogee, and height above bottom.

    Height above apogee is calculated independently for each profile:

        height_above_apogee = max(depth in profile) - depth

    Height above bottom is:

        height_above_bottom = -(depth + bathymetry)

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing DEPTH, PROFILE_NUMBER, LONGITUDE and LATITUDE.

    bathymetry : xarray.Dataset
        GEBCO bathymetry dataset containing 'elevation',
        'lon', and 'lat'.

    Returns
    -------
    xarray.Dataset
        Dataset with BATHYMETRY, HEIGHT_ABOVE_APOGEE and
        HEIGHT_ABOVE_BOTTOM added.
    """

    # --------------------------------------------------
    # 1. Get bathymetry at each measurement position
    # --------------------------------------------------
    if bathymetry is not None:
        elevation = bathymetry["elevation"].sel(
            lon=ds["LONGITUDE"],
            lat=ds["LATITUDE"],
            method="nearest",
        ).values

        ds["HEIGHT_ABOVE_BOTTOM"] = -(
                ds["DEPTH"] + elevation
            )

    # --------------------------------------------------
    # 2. Height above apogee
    # --------------------------------------------------
    depth = ds["DEPTH"].values
    profile_n = ds["PROFILE_NUMBER"].values

    height_above_apogee = np.full(
        depth.shape,
        np.nan,
        dtype=float,
    )

    for profile in np.unique(profile_n):
        mask = profile_n == profile

        depth_profile = depth[mask]

        if np.all(np.isnan(depth_profile)):
            continue

        apogee_depth = np.nanmax(depth_profile)

        height_above_apogee[mask] = (
            apogee_depth - depth_profile
        )

    ds["HEIGHT_ABOVE_APOGEE"] = (
        ds["DEPTH"].dims,
        height_above_apogee,
    )
    return ds
