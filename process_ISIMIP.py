import os
import glob
import xarray as xr
import numpy as np
import logging
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from typing import Any, Dict, List, Tuple
import os, uuid, time, gc

def merge_ISIMIP_datasets(src_isimip, models, scenarios, variables, overwrite=False, verbose=False, log_file='merge_isimip.log'):
    """
    Merges ISIMIP datasets for specified models, scenarios, and climate variables.

    Parameters:
        src_isimip (str): Source directory containing ISIMIP data to be merged.
        models (list): List of model names.
        scenarios (list): List of scenario names.
        variables (list): List of variable names to merge.
        overwrite (bool): Whether to overwrite existing merged files.
        verbose (bool): Whether to print progress messages to console.
        log_file (str): Path to log file.

    Returns:
        dict: Summary of merged files.
    """

    # Set up logging
    logger = logging.getLogger('ISIMIP_Merger')
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers on repeated calls
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if verbose else logging.WARNING)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers
        logger.addHandler(fh)
        logger.addHandler(ch)

    summary = {}

    for model in models:
        for scenario in scenarios:
            for variable in variables:
                key = (model, scenario, variable)
                files = sorted(glob.glob(f'{src_isimip}/{model}/{scenario}/{model}*_{variable}_*.nc'))
                summary[key] = {'files_found': len(files), 'merged': False}

                if not files:
                    logger.warning(f'No files found for {key}')
                    continue

                output_file = f'{src_isimip}/{model}/{scenario}/{model}_{variable}_merged.nc'

                if os.path.exists(output_file) and not overwrite:
                    logger.info(f'Skipping existing file: {output_file}')
                    continue

                try:
                    logger.info(f'Merging {len(files)} files for {key}')
                    with xr.open_mfdataset(files, combine='by_coords', parallel=True) as ds:
                        ds.to_netcdf(output_file)

                    summary[key]['merged'] = True
                    logger.info(f'Successfully saved: {output_file}')

                except Exception as e:
                    summary[key]['error'] = str(e)
                    logger.error(f'Error processing {key}: {e}')

    return summary
#=========================================================================================================================
#REGRID ISIMIP DATASETS with XESMF

#--------------------------------------------
#Define the EOBS dataset and variable
def select_ref_hist(dataset, var, start_date, end_date):

    """
    Selects the historical data for a specific variable and time period and masks out cells with low variance and removes empty rows and columns.
    """

    #drop zero variance grids
    dataset= dataset[var].sel(time=slice(start_date, end_date)).where(dataset[var].var(dim='time') > 0.0001)

    #drop NaN columns and rows
    dataset = dataset.where(dataset.notnull().any(dim='time')).dropna(dim='lat', how='all').dropna(dim='lon', how='all')
    return dataset
#--------------------------------------------

def regrid_ISIMIP_to_obs(isimip_data: dict, obs: xr.DataArray, models: list, scenarios: list, REGRID_METHOD: str,
                          VAR_NAME: str, VAR_UNITS: str, VAR_STDNAME: str, long_name: str) -> dict:  
    """
    Regrid ISIMIP data to EOBS grid using XESMF and return a dictionary of model-scenario combinations.
    Each key in the dictionary is a string formatted as 'model_scenario_variable', and the value is the regridded data as an xarray.DataArray.

    Parameters:
        ----------
    isimip_data (dict): Dictionary containing ISIMIP data.
    obs (xarray.Dataset): EOBS dataset to regrid to.
    models (list): List of model names.
    scenarios (list): List of scenario names.
    METHOD (str): Regridding method to use (e.g., 'bilinear').
    VAR_NAME (str): Name of the variable to regrid (e.g., 'pr', 'tasmin', 'tasmax').
    VAR_UNITS (str): Units of the variable.
    VAR_STDNAME (str): Standard name of the variable.
    long_name (str): Long name of the variable.


    
    Returns:
    dict: Dictionary with model-scenario combinations as keys and regridded data as values.
    """
        
    model_scenario_combos ={}

    #create regridder from one of the models. Only if all models have the same extent and resolution
    #data_grid = xe.Regridder(isimip_data[(models[0], scenarios[0], VAR_NAME)], obs, method='bilinear')


    for model in models:
        for scenario in scenarios:
                if f'{model}_{scenario}_{VAR_NAME}' in isimip_data:

                    #change key to model_scenario_pr
                    name = f"{model}_{scenario}_{VAR_NAME}"

                    # Extract the variable
                    if VAR_NAME == 'pr':
                        # Convert precipitation flux from kg m-2 s-1 to mm/day
                        data = isimip_data[f'{model}_{scenario}_{VAR_NAME}'] * 86400

                        #If temperature variables, convert from Kelvin to Celsius
                    elif VAR_NAME in ['tasmin', 'tasmax']:
                        data = isimip_data[f'{model}_{scenario}_{VAR_NAME}']-273.15

                    #regrid the data to the EOBS grid with xesmf
                    #regrid with xesmf
                    data_grid = xe.Regridder(data, obs, method=REGRID_METHOD)

                    # Apply the regridder to the historical simulation
                    data_regrid = data_grid(data)

                    #Make a mask where obs is not null
                    valid_mask = obs.notnull().any(dim="time")

                    #data where variance is not zero
                    #resample to daily
                    data_regrid = data_regrid.resample(time='1D').sum()
                    
                    data_regrid = data_regrid.where(data_regrid.var(dim='time') > 0.0001)
                    data_regrid = data_regrid.where(valid_mask)


                    #Define atributes
                        # 5) Attach useful attributes
                    attrs = dict(
                        units=VAR_UNITS,
                        standard_name=VAR_STDNAME,
                        long_name= long_name,
                        source="ISIMIP3b",
                        source_model=model,
                        source_scenario=scenario,
                        regrid_method=REGRID_METHOD,
            )

                    #assign attributes
                    data_regrid = data_regrid.assign_attrs(**attrs)

                    #add to dictionary
                    model_scenario_combos[name] = data_regrid

    
    return model_scenario_combos


#=========================================================================================================================
"""
BIAS-CORRECTION_SCRIPTS
"""
#=========================================================================================================================
#EMPIRICAL QUANTILE MAPPING (EQM) for bias correction of precipitation-like variables

#=========================================================================================================================
#This script implements a two-pass EQM bias correction method with zero-jittering for precipitation-like variables.
#It is designed to work with xarray DataArrays and can be applied to historical and future climate model outputs.
#The EQM method adjusts the distribution of model outputs to match observed data, improving the realism of climate projections.
#=========================================================================================================================

# ── Helpers (zero-only jitter + EQM) ─────────────────────────────────────────

def jitter_zeros(arr, jitter=1e-3):
    """
    Add small random noise to zero values only, then clip negatives to 0.

    Why: Precipitation series often have many zeros. Without jitter, the model CDF
    has flat steps at 0, which can cause unstable/degenerate quantile mapping.
    Jitter keeps zeros near zero while making the CDF strictly increasing.

    Parameters
    ----------
    arr : np.ndarray
        Input array (NaNs allowed).
    jitter : float
        Max absolute noise added to zeros (uniform in [-jitter, +jitter]).

    Returns
    -------
    np.ndarray (float64)
        Array with zero entries slightly perturbed, clipped to non-negative.
    """
    out = arr.astype(np.float64, copy=True)   # ensure float for NaNs + noise
    zeros = (out == 0)
    if np.any(zeros):
        noise = np.random.uniform(-jitter, jitter, size=out.shape)
        out[zeros] += noise[zeros]
        out[out < 0] = 0.0  # ensure non-negative (precip)
    return out


def compute_eqm_mapping(o, h, n_q=51, min_samples=10):
    """
    Build the empirical quantile mapping for one cell-month.

    We compute quantiles for the observed series (qo) and for the historical
    model series (qh, after jittering zeros). Later, for any x we find an
    adjustment factor af by interpolating the ratio qo/qh at the x value.

    Parameters
    ----------
    o : np.ndarray
        Observed values for a single month & cell (1-D: time).
    h : np.ndarray
        Historical modeled values for same month & cell (1-D: time).
    n_q : int
        Number of quantiles (e.g., 51 → 0, 0.02, …, 1.0).
    min_samples : int
        Minimum #valid (non-NaN) pairs to calibrate; otherwise return None.

    Returns
    -------
    tuple | None
        (qh, qo) quantile arrays (same shape), or None if insufficient data.
    """
    valid = (~np.isnan(o)) & (~np.isnan(h))
    if np.count_nonzero(valid) < min_samples:
        return None

    probs = np.linspace(0, 1, n_q)           # common probs grid
    h_j = jitter_zeros(h)                     # jitter only the model side
    qh = np.quantile(h_j[valid], probs)       # model quantiles
    qo = np.quantile(o [valid], probs)        # observed quantiles
    return qh, qo


def apply_eqm_vec(x, qh, qo):
    """
    Apply ratio EQM to a 1-D time series x using precomputed (qh, qo).

    Steps:
      1) Build multiplicative factors af = qo/qh (where qh>0; else use 1).
      2) Interpolate af as a function of x over the support defined by qh.
      3) Scale x by the interpolated factor; keep structural zeros as zeros.

    Parameters
    ----------
    x : np.ndarray
        Series to correct (1-D).
    qh : np.ndarray
        Historical model quantiles (from calibration).
    qo : np.ndarray
        Observed quantiles (from calibration).

    Returns
    -------
    np.ndarray
        Corrected series, with zeros preserved.
    """
    # Initialize multiplicative factors as ones; fill positive qh with ratios
    af = np.ones_like(qh, dtype=np.float64)
    nz = (qh > 0)
    af[nz] = qo[nz] / qh[nz]

    # Interpolate factor in x-space; clamp outside to edge factors
    corr = np.interp(x, qh, af, left=af[0], right=af[-1])

    # Apply correction; preserve structural zeros explicitly
    y = (x * corr).astype(np.float64, copy=False)
    y[x == 0] = 0.0
    return y


# ── Main two-pass function ───────────────────────────────────────────────────

def empirical_quantile_mapping(obs, hist, fut, n_q=51, min_samples=10):
    """
    Bias-correct historical and future modeled data using ratio-based EQM.

    Phase A: TRAIN
      - Align obs & hist on time (inner join).
      - For each calendar month and each (lat, lon) cell:
          * compute (qh, qo) quantiles and store mapping.

    Phase B: APPLY
      - For each month:
          * apply the corresponding cell-level mapping to both hist and fut.

    Parameters
    ----------
    obs : xr.DataArray   [time, lat, lon]
        Observed reference data.
    hist : xr.DataArray  [time, lat, lon]
        Historical model data (same grid as obs).
    fut : xr.DataArray   [time, lat, lon]
        Future model data (same grid as obs/hist) — *must be regridded beforehand*.
    n_q : int
        Number of quantiles used to build the mapping.
    min_samples : int
        Minimum #valid samples (per cell-month) to fit a mapping.

    Returns
    -------
    hist_bc : xr.DataArray (float64)
        Bias-corrected historical series on the original hist grid/time.
    fut_bc  : xr.DataArray (float64)
        Bias-corrected future series on the original fut grid/time.

    Raises
    ------
    ValueError
        If obs/hist/fut are not on the same spatial grid (lat/lon).
    """

    # A1) Align obs & hist on time to ensure paired samples for training
    obs_a, hist_a = xr.align(obs, hist, join="inner")

    # A2) Sanity check: spatial grids must match across all inputs
    if not (np.array_equal(obs.lat, hist.lat) and np.array_equal(obs.lon, hist.lon)):
        raise ValueError("obs and hist must share identical lat/lon grids.")
    if not (np.array_equal(obs.lat, fut.lat) and np.array_equal(obs.lon, fut.lon)):
        raise ValueError("fut must be on the same lat/lon grid as obs/hist (regrid first).")

    # B) Pre-allocate float outputs so NaNs are supported even if inputs were int
    hist_bc = xr.full_like(hist, np.nan, dtype=np.float64)
    fut_bc  = xr.full_like(fut,  np.nan, dtype=np.float64)

    lats, lons = obs.lat.values, obs.lon.values

    # A3) TRAIN: precompute month×cell mappings
    #     Key: (month, i, j) → Value: (qh, qo)
    mappings = {}
    for month in range(1, 13):
        # mask the aligned training time axis by calendar month
        msk = obs_a.time.dt.month == month
        if not msk.any():
            continue  # this dataset has no timestamps for this month

        # Extract month slices as NumPy (eager; fine for medium datasets)
        o_m = obs_a.sel(time=msk).values  # shape: (t_m, nlat, nlon)
        h_m = hist_a.sel(time=msk).values

        # Loop cells; micro-opt: slice rows once to reduce Python overhead
        for i in range(len(lats)):
            o_row = o_m[:, i, :]
            h_row = h_m[:, i, :]
            for j in range(len(lons)):
                res = compute_eqm_mapping(o_row[:, j], h_row[:, j],
                                          n_q=n_q, min_samples=min_samples)
                if res is not None:
                    mappings[(month, i, j)] = res  # (qh, qo)

    # B1) APPLY: use trained mappings month-wise to correct hist and fut
    for month in range(1, 13):
        msk_hist = (hist.time.dt.month == month).values
        msk_fut  = (fut .time.dt.month == month).values
        if not (msk_hist.any() or msk_fut.any()):
            continue

        # Pull raw arrays for speed (note: eager; consider dask-vectorization for big data)
        if msk_hist.any():
            h_all = hist.values[msk_hist, ...]  # (t_hm, nlat, nlon)
        if msk_fut.any():
            f_all = fut.values[msk_fut,  ...]   # (t_fm, nlat, nlon)

        for i in range(len(lats)):
            for j in range(len(lons)):
                key = (month, i, j)
                if key not in mappings:
                    # Insufficient training samples for this cell-month → leave NaNs
                    continue

                qh, qo = mappings[key]

                if msk_hist.any():
                    h_bc = apply_eqm_vec(h_all[:, i, j], qh, qo)
                    hist_bc.values[msk_hist, i, j] = h_bc

                if msk_fut.any():
                    f_bc = apply_eqm_vec(f_all[:, i, j], qh, qo)
                    fut_bc.values[msk_fut,  i, j] = f_bc

    # Nice-to-have: carry over metadata if present
    hist_bc.name = getattr(hist, "name", "hist_bc")
    fut_bc.name  = getattr(fut,  "name",  "fut_bc")
    hist_bc.attrs.update(getattr(hist, "attrs", {}))
    fut_bc.attrs.update(getattr(fut,  "attrs", {}))

    return hist_bc, fut_bc

#==========================================================================================================================
"""
Quantile Delta Mapping (QDM) over a lat×lon grid (xarray) using Joblib parallelism.
Includes:
  • DataFrame‑based QDM helpers (compute CDFs, apply to FUT/HIST)
  • A per‑cell worker that runs QDM for one (lat_j, lon_i)
  • A grid driver that stitches results back into xarray DataArrays

Choose kind="*" for precipitation‑like variables (multiplicative),
and kind="+" for temperature‑like variables (additive).

──────────────────────────────────────────────────────────────────────────────
                             HIGH‑LEVEL WORKFLOW
──────────────────────────────────────────────────────────────────────────────
Inputs
  obs(time,lat,lon), sim_hist(time,lat,lon), sim_future(time,lat,lon)

Train (implicit inside helpers via time intersection)
  For each grid cell (j,i):
    1) Align obs & hist on overlapping time
    2) Build probs ps and quantiles q_obs, q_simh
    3) QDM FUT at ps → corrected FUT series
    4) QM/QDM HIST at ps → corrected HIST series (diagnostics)

Apply
  Paste corrected 1‑D arrays into (lat,lon,time) numpy cubes
  Convert to xr.DataArray → hist_bc_da, fut_bc_da

Notes
  • This version uses pandas per‑cell; for very large grids consider a vectorized
    xarray/dask path to avoid Python overhead.
  • FUT must already be on the same lat/lon grid as OBS/HIST.
  • Extrapolation at the tails uses edge values of ps (0 or 1).
──────────────────────────────────────────────────────────────────────────────
"""
#==========================================================================================================================
import numpy as np
import xarray as xr

def quantile_delta_mapping(
    obs: xr.DataArray,
    sim_hist: xr.DataArray,
    sim_future: xr.DataArray,
    *,                          #means keyword-only arguments after this point i.e. n_quantiles=xxx not just typing xxx
    n_quantiles: int = 251,
    min_valid: int = 10,
    kind: str = "+",          # "+" (additive, e.g., T) or "*" (multiplicative, e.g., P)
    ps_eps: float = 1e-3,     # shrink tails away from 0/1 for stability
    eps: float = 1e-6,        # floor for multiplicative denominator
    rmin: float = 0.2,        # min scaling factor (multiplicative)
    rmax: float = 10.0,       # max scaling factor (multiplicative)
    out_dtype: np.dtype = np.float32,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Overlap-only QDM (vectorized, dask-friendly).
    - Calibrates on obs×sim_hist time overlap.
    - Returns:
        hist_bc: QM-corrected sim_hist on overlap times
        fut_bc : QDM-corrected sim_future on its native times
    """

    if kind not in {"+", "*"}:
        raise ValueError("kind must be '+' or '*'")

    # Grid sanity (expects dims named 'lat','lon')
    if not (np.array_equal(obs.lat, sim_hist.lat) and np.array_equal(obs.lon, sim_hist.lon)):
        raise ValueError("obs and sim_hist must share the same lat/lon grid.")
    if not (np.array_equal(obs.lat, sim_future.lat) and np.array_equal(obs.lon, sim_future.lon)):
        raise ValueError("sim_future must share the same lat/lon grid as obs/sim_hist.")

    # 1) Align training on overlap
    obs_aln, hist_aln = xr.align(obs, sim_hist, join="inner")

    # Stack/unstack helpers
    def _stack(da): return da.transpose("time", "lat", "lon").stack(space=("lat", "lon"))
    def _unstack(da): return da.unstack("space").transpose("time", "lat", "lon")

    O = _stack(obs_aln)            # (t_train, space)
    Ht = _stack(hist_aln)          # (t_train, space)
    F  = _stack(sim_future)        # (t_fut,   space)

    # 2) Probability grid (avoid exact 0/1)
    ps = xr.DataArray(
        np.linspace(ps_eps, 1.0 - ps_eps, n_quantiles, dtype=np.float64),
        dims=["q"], name="ps"
    )

    # 3) Quantiles per space (nan-aware; require min_valid)
    def _nanquantile_1d(x, q):
        x = x.astype(np.float64, copy=False)
        if np.count_nonzero(~np.isnan(x)) < min_valid:
            return np.full(q.shape, np.nan, dtype=np.float64)
        return np.nanquantile(x, q)

    q_obs  = xr.apply_ufunc(_nanquantile_1d, O, ps,
                            input_core_dims=[["time"], ["q"]], output_core_dims=[["q"]],
                            vectorize=True, dask="parallelized", output_dtypes=[np.float64])
    q_simh = xr.apply_ufunc(_nanquantile_1d, Ht, ps,
                            input_core_dims=[["time"], ["q"]], output_core_dims=[["q"]],
                            vectorize=True, dask="parallelized", output_dtypes=[np.float64])
    q_simf = xr.apply_ufunc(_nanquantile_1d, F, ps,
                            input_core_dims=[["time"], ["q"]], output_core_dims=[["q"]],
                            vectorize=True, dask="parallelized", output_dtypes=[np.float64])

    # 3b) Enforce non-decreasing quantiles for robust interp
    def _monotone(q):
        if np.any(np.isnan(q)): return q
        return np.maximum.accumulate(q)

    q_obs  = xr.apply_ufunc(_monotone, q_obs,  input_core_dims=[["q"]], output_core_dims=[["q"]],
                            vectorize=True, dask="parallelized", output_dtypes=[np.float64])
    q_simh = xr.apply_ufunc(_monotone, q_simh, input_core_dims=[["q"]], output_core_dims=[["q"]],
                            vectorize=True, dask="parallelized", output_dtypes=[np.float64])
    q_simf = xr.apply_ufunc(_monotone, q_simf, input_core_dims=[["q"]], output_core_dims=[["q"]],
                            vectorize=True, dask="parallelized", output_dtypes=[np.float64])

    # 4) Interp helpers
    def _p_from_q(x_t, qx_q, ps_q):
        if np.any(np.isnan(qx_q)): return np.full_like(x_t, np.nan, dtype=np.float64)
        return np.interp(x_t, qx_q, ps_q, left=ps_q[0], right=ps_q[-1])

    def _x_from_p(p_t, ps_q, qo_q):
        if np.any(np.isnan(qo_q)): return np.full_like(p_t, np.nan, dtype=np.float64)
        return np.interp(p_t, ps_q, qo_q)

    # 5) HIST (QM) on overlap only
    p_hist = xr.apply_ufunc(_p_from_q, Ht, q_simh, ps,
                            input_core_dims=[["time"], ["q"], ["q"]],
                            output_core_dims=[["time"]], vectorize=True,
                            dask="parallelized", output_dtypes=[np.float64])

    H_bc = xr.apply_ufunc(_x_from_p, p_hist, ps, q_obs,
                          input_core_dims=[["time"], ["q"], ["q"]],
                          output_core_dims=[["time"]], vectorize=True,
                          dask="parallelized", output_dtypes=[out_dtype])

    # 6) FUT (QDM)
    def _corr_quantiles(obs_q, simh_q, simf_q, mode, eps_, rmin_, rmax_):
        if np.any(np.isnan(obs_q)) or np.any(np.isnan(simh_q)) or np.any(np.isnan(simf_q)):
            return np.full_like(obs_q, np.nan, dtype=np.float64)
        if mode == "+":
            return obs_q + (simf_q - simh_q)
        simh_q_safe = np.maximum(simh_q, eps_)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = simf_q / simh_q_safe
        ratio = np.clip(ratio, rmin_, rmax_)
        return obs_q * ratio

    corr_q = xr.apply_ufunc(_corr_quantiles, q_obs, q_simh, q_simf,
                            xr.DataArray(kind), xr.DataArray(eps),
                            xr.DataArray(rmin), xr.DataArray(rmax),
                            input_core_dims=[["q"], ["q"], ["q"], [], [], [], []],
                            output_core_dims=[["q"]], vectorize=True,
                            dask="parallelized", output_dtypes=[np.float64])

    p_fut = xr.apply_ufunc(_p_from_q, F, q_simf, ps,
                           input_core_dims=[["time"], ["q"], ["q"]],
                           output_core_dims=[["time"]], vectorize=True,
                           dask="parallelized", output_dtypes=[np.float64])

    F_bc = xr.apply_ufunc(_x_from_p, p_fut, ps, corr_q,
                          input_core_dims=[["time"], ["q"], ["q"]],
                          output_core_dims=[["time"]], vectorize=True,
                          dask="parallelized", output_dtypes=[out_dtype])

    # 7) Unstack & name; set coords
    hist_bc = _unstack(H_bc).assign_coords(time=hist_aln.time).rename("hist_qdm")
    fut_bc  = _unstack(F_bc).assign_coords(time=sim_future.time).rename("future_qdm")

    # Carry units if present
    if "units" in getattr(obs, "attrs", {}):
        hist_bc.attrs.setdefault("units", obs.attrs["units"])
        fut_bc .attrs.setdefault("units", obs.attrs["units"])

    return hist_bc, fut_bc

#==========================================================================================================================

### bias correct the regridded data
def bias_correct_ISIMIP(isimip_regridded: dict, obs: xr.DataArray, models: list, 
                        future_scenarios: list, method: str,
                        VAR_NAME, VAR_UNITS, VAR_STDNAME, kind):
    """
    Bias-correct regridded ISIMIP data using QDM or EQM (overlap-only training).
    Returns a dict with bias-corrected historical and future DataArrays per model/scenario.
    """
    method_norm = method.lower()
    if method_norm not in {"qdm", "eqm"}:
        raise ValueError("Method must be 'QDM' or 'EQM'.")

    # dict to hold output DataArrays
    out = {}

    for model in models:
        key_hist = f"{model}_historical_{VAR_NAME}"
        hist_entry = isimip_regridded.get(key_hist)
        if hist_entry is None or VAR_NAME not in hist_entry:
            print(f"[skip] missing {key_hist}")
            continue
        hist_data = hist_entry[VAR_NAME]

        # crop obs once per model
        obs_sub = obs.sel(lat=slice(hist_data.lat.max(), hist_data.lat.min()),
                          lon=slice(hist_data.lon.min(), hist_data.lon.max()))

        for future in future_scenarios:
            key_fut = f"{model}_{future}_{VAR_NAME}"
            fut_entry = isimip_regridded.get(key_fut)
            if fut_entry is None or VAR_NAME not in fut_entry:
                print(f"[skip] missing {key_fut}")
                continue
            fut_data = fut_entry[VAR_NAME]

            # (optional but helpful) ensure same grid between hist and fut
            if not (np.array_equal(hist_data.lat, fut_data.lat) and np.array_equal(hist_data.lon, fut_data.lon)):
                raise ValueError(f"{model} {future}: future grid != historical grid after regridding.")

            if method_norm == "qdm":
                hist_bc, fut_bc = quantile_delta_mapping(
                    obs_sub, hist_data, fut_data, n_quantiles=251, min_valid=10, kind=kind
                )
            else:  # eqm
                hist_bc, fut_bc = empirical_quantile_mapping(
                    obs_sub, hist_data, fut_data, n_q=51, min_samples=10
                )

            long_name = f"{method_norm.upper()} bias-corrected {VAR_NAME}"

            # variable-level attributes
            var_attrs = {
                "units": VAR_UNITS,
                "standard_name": VAR_STDNAME,
                "long_name": long_name,
            }

            # dataset-level attributes
            hist_attrs = {
                "source": f"{model} ISIMIP3b",
                "source_scenario": "historical",
                "bias_correction_method": method_norm.upper()
                
            }
            fut_attrs = {
                "source": f"{model} ISIMIP3b",
                "source_scenario": future,
                "bias_correction_method": method_norm.upper()
                
            }

            # build datasets
            hist_ds = xr.Dataset({VAR_NAME: hist_bc.astype(np.float32)})
            hist_ds[VAR_NAME].attrs = var_attrs
            hist_ds.attrs.update(hist_attrs)

            fut_ds = xr.Dataset({VAR_NAME: fut_bc.astype(np.float32)})
            fut_ds[VAR_NAME].attrs = var_attrs
            fut_ds.attrs.update(fut_attrs)

            # save datasets to results dict
            out[f"{model}_historical_{VAR_NAME}"] = hist_ds
            out[f"{model}_{future}_{VAR_NAME}"]   = fut_ds

    return out

#==========================================================================================================================


#====Export  bias corrected data to NetCDF

def export_scenarios_to_netcdf(isimip_dict, dest_isimip: str, export_scenarios: list,
                               models: list, variable: str) -> None:
    """
    Export the bias-corrected ISIMIP data to NetCDF files, safely handling file locks on Windows.
    """
    
    for model in models:
        for scenario in export_scenarios:
            data = isimip_dict[f'{model}_{scenario}_{variable}']
            # Load fully into memory to detach from any open NetCDF source
            data = data.load()
            # Ensure target folder exists
            out_dir = os.path.join(dest_isimip, model, scenario)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"{model}_{variable}.nc")

            # Temp file for atomic write
            tmp_file = os.path.join(out_dir, f".tmp_{uuid.uuid4().hex}.nc")

            # Try a few times in case file is locked
            for attempt in range(5):
                try:
                    data.to_netcdf(tmp_file, mode='w', format='NETCDF4')
                    gc.collect()
                    os.replace(tmp_file, out_file)  # Atomic replace
                    break
                except PermissionError:
                    time.sleep(0.6 * (attempt + 1))
            else:
                raise PermissionError(f"Could not write file {out_file}. "
                                        f"Ensure no other program has it open.")
