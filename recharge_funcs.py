#THese scripts are for validating mHM flow and baseflow for seasonal recharge analysis
"""Python py_geospatial environment"""
import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
import process_observed_discharge as mQ
import glob
from sklearn.metrics import r2_score
import baseflow #this is the baseflow package
import sys
from pathlib import Path
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
import geopandas as gpd
from scipy import stats
#segoe UI
plt.rcParams['font.family'] = 'Segoe UI'
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


##function to match frequency and resample to desired frequency
def match_frequency_and_resample(q_model: pd.Series, q_obs: pd.Series, obs_freq: str, resample_freq: str, station_id: str):
    """
    Match frequency of model and observed discharge time series.
    Resample both to the specified frequency if they differ.
    Parameters
    ----------
    q_model : pd.Series  
        Modeled discharge time series (indexed by date).  
    q_obs : pd.Series  
        Observed discharge time series (indexed by date).  
    obs_freq : str  
        The frequency of observed data (e.g., 'D' for daily).  
    resample_freq : str  
        Resampling frequency (e.g., 'ME' for month-end).  
    station_id : str  
        Identifier for the station (for logging purposes).  
    Returns
    -------
    pd.Series, pd.Series  
        Resampled model and observed discharge time series.  
    """
    #Check if both series have the same frequency and resample if not
    #Check if both series have the same frequency and resample if not
    #set indices to datetime if not already
    if not isinstance(q_model.index, pd.DatetimeIndex):
        q_model.index = pd.to_datetime(q_model.index)
    if not isinstance(q_obs.index, pd.DatetimeIndex):
        q_obs.index = pd.to_datetime(q_obs.index)

    #Enforce daily frequency for Observed data if not already
    q_obs = q_obs.asfreq(obs_freq)   # enforce daily frequency


    if not (pd.infer_freq(q_model.index) == pd.infer_freq(q_obs.index)):
        print(f"Station {station_id}: Data will be resampled to {resample_freq}.", end="\r")

        q_model_f = q_model.resample(resample_freq).mean()

        #resample observed data only if each month has at least 15 days of data
        q_obs_f = (
            q_obs
            .resample(resample_freq)
            .apply(lambda x: x.mean() if x.count() >= 20 else np.nan)
        )

    else:
        q_model_f = q_model
        q_obs_f = q_obs

    # Align and drop NaNs
    merged_Q = pd.concat([q_model_f, q_obs_f], axis=1)
    merged_Q.columns = ["q_model", "q_obs"]
    merged_Q = merged_Q.dropna()

    return merged_Q

#==================================================

#Function to extract per station flow quantiles

def extract_flow_quantiles(merged_Q: pd.Series, station_id: str):
    """
    Extract Q50 and Q90 for model vs. observed discharge at a given station.
    Returns a single-row DataFrame or None if too short.
    Parameters
    ----------
    merged_Q : pd.Series
        DataFrame with two columns: 'q_model' and 'q_obs', indexed by date.
    station_id : str
        Identifier for the station (for logging purposes).
    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ['station', 'q50_model', 'q90_model', 'q50_obs', 'q90_obs']
        or None if insufficient overlapping data.
    """
   
    # If DataFrame, reduce to first column
    if isinstance(merged_Q, pd.DataFrame):
        q_model = merged_Q.iloc[:, 0]
        q_obs = merged_Q.iloc[:, 1]

    if len(merged_Q) > 12:  # at least 2 years of overlapping data
        print(f"Station {station_id}: {len(merged_Q)} overlapping data points found.", end="\r")
        q50_model = np.nanpercentile(q_model.values, 50)
        q90_model = np.nanpercentile(q_model.values, 10)
        q50_obs   = np.nanpercentile(q_obs.values, 50)
        q90_obs   = np.nanpercentile(q_obs.values, 10)

        return pd.DataFrame(
            {
                "station": [station_id],
                "q50_model": [q50_model],
                "q90_model": [q90_model],
                "q50_obs": [q50_obs],
                "q90_obs": [q90_obs],
            }
        )
    else:
        print(f"Station {station_id}: skipped ({len(q_model)} points only).", end="\r")
        return None

#==================================================
#change dictionary keys to upper case
def keys_upper(test_dict):
    res = dict()
    for key in test_dict.keys():
        if isinstance(test_dict[key], dict):
            res[key.upper()] = keys_upper(test_dict[key])
        else:
            res[key.upper()] = test_dict[key]
    return res

#==================================================
#Function to extract flow quantiles for all stations and models

def extract_station_quantiles(base_sim_dir, sim_subfolder, models, eval_Obs):
    """  
    Extract flow quantiles (Q50 and Q90) for each station and model.  
    Parameters  
    ----------
    base_sim_dir : str  
        Base directory containing model subdirectories.  
    sim_subfolder : str  
        Subfolder within each model directory containing simulation CSV files.  
    models : list of str  
        List of model names corresponding to subdirectory names.  
    eval_Obs : dict  
        Dictionary of observed discharge time series (keyed by station name).    
    Returns  
    -------
    pd.DataFrame  
        DataFrame with columns ['station', 'q50_model', 'q90_model', 'q50_obs', 'q90_obs', 'model']  
    """

    all_quantiles = {}

    for model in models:
        sim_files = glob.glob(f"{base_sim_dir}/{model}/{sim_subfolder}/*.csv")
        quantiles = []  # <-- collect per model

        for fpath in sim_files:
            station_name = os.path.splitext(os.path.basename(fpath))[0]

            #match cases
            station_name = station_name.upper()

            #change dict keys to upper case
            eval_Obs_upper = keys_upper(eval_Obs)
            if station_name in eval_Obs_upper.keys():

                obs_Q = eval_Obs_upper[station_name]
                #set index to datetime if not already
                if not isinstance(obs_Q.index, pd.DatetimeIndex):
                    obs_Q.index = pd.to_datetime(obs_Q.index)
                sim_Q = pd.read_csv(fpath, index_col=0, parse_dates=True)

                # Match frequency and resample to monthly
                q_merged = match_frequency_and_resample(sim_Q, obs_Q, 'D','ME', station_name)

                qn = extract_flow_quantiles(q_merged, station_name)
                if qn is not None:
                    qn["model"] = model
                    quantiles.append(qn)

            if quantiles:  # <-- use the correct list
                quantiles_df = pd.concat(quantiles, ignore_index=True)
            else:
                quantiles_df = pd.DataFrame()

            all_quantiles[model] = quantiles_df
        
    #save to df
    all_quantiles_df = pd.concat(all_quantiles.values(), ignore_index=True)

    return all_quantiles_df

#==================================================

def seasonal_Q_comparison(base_sim_dir, models, eval_Obs, season_map):
    """  
    Extract seasonal (DJF, MAM, JJA, SON) mean discharge for each station and model.  
    Parameters  
    ----------
    models : list of str  
        List of model names corresponding to subdirectory names.  
    eval_Obs : dict  
        Dictionary of observed discharge time series (keyed by station name).    

    season_map : dict
        Mapping of month numbers to season labels.  

    Returns  
    -------
    dict of pd.DataFrame  
        Dictionary with model names as keys and DataFrames with seasonal means as values.  
    """

    all_seasonal = {}

    for model in models:
        sim_files = glob.glob(f"{base_sim_dir}/{model}/Qrouted/*.csv")
        model_season = []

        for fpath in sim_files:
            station_name = os.path.splitext(os.path.basename(fpath))[0]

            #match cases
            station_name = station_name.upper()

            #change dict keys to upper case
            eval_Obs_upper = keys_upper(eval_Obs)

            if station_name in eval_Obs_upper.keys():

                obs_Q = eval_Obs_upper[station_name]
                sim_Q = pd.read_csv(fpath, index_col=0, parse_dates=True)

                # merge by date
                merged_df = pd.concat([sim_Q, obs_Q], axis=1, join="inner")
                merged_df.columns = ["q_model", "q_obs"]
                merged_df = merged_df.dropna()

                # add season label
                merged_df["season"] = merged_df.index.month.map(season_map)

                # seasonal climatology
                season_means = merged_df.groupby("season")[["q_model","q_obs"]].mean().reset_index()
                season_means["name"] = station_name
                season_means["model"] = model

                model_season.append(season_means)

            if model_season:
                all_seasonal[model] = pd.concat(model_season, ignore_index=True)
    
    model_seasons_df = pd.concat(all_seasonal.values(), ignore_index=True)
    model_seasons_df = model_seasons_df[["name", "model", "season", "q_obs", "q_model"]]

    return model_seasons_df

#==================================================
def extract_multistation_baseflow(base_sim_dir, models, eval_Obs):
    """  
    Extract baseflow time series for each station and model using the best method based on observed data.  
    Parameters  
    ----------
    base_sim_dir : str
        Base directory containing model subdirectories.
    models : list of str
        List of model names corresponding to subdirectory names.
    eval_Obs : dict  
        Dictionary of observed discharge time series (keyed by station name).    
    Returns  
    -------
    dict of pd.DataFrame  
        Dictionary with model names as keys and DataFrames with baseflow time series as values.
    """
    # Ensure station names are uppercase for consistency
    eval_Obs_upper = keys_upper(eval_Obs)

    all_models_Qb = {}
    all_models_BFI = {}
    best_BF_method = {}

    for model in models:
        sim_dir = Path(base_sim_dir) / model / 'Qrouted'
        stations_Qb = []
        stations_BFI = []

        for fpath in sorted(sim_dir.glob("*.csv")):
            station_name = fpath.stem.upper()

            if station_name not in eval_Obs_upper:
                #extract baseflow
                msg = f"Extracting baseflow for {station_name}"
                sys.stdout.write("\r" + msg + "  " * 20)
                sys.stdout.flush()
                continue

            obs_Q = eval_Obs_upper[station_name]
            #set index to datetime if not already
            if not isinstance(obs_Q.index, pd.DatetimeIndex):
                obs_Q.index = pd.to_datetime(obs_Q.index)
            sim_Q = pd.read_csv(fpath, index_col=0, parse_dates=True)

            #merge on index
            q_merged = match_frequency_and_resample(sim_Q, obs_Q, 'D','D', station_name)

            if q_merged.empty:
                continue
            
            #extract baseflow
            msg = f"Extracting baseflow for {station_name}"
            sys.stdout.write("\r" + msg + "  " * 20)
            sys.stdout.flush()

            #extract baseflow using multiple methods and select best based on KGE
            obs_bf_dict, obs_bfi, obs_kge = baseflow.separation(q_merged[["q_obs"]], return_bfi=True, return_kge=True)

            #select the best method based on KGE
            best_method = obs_kge.idxmax(axis=1).iloc[0]  #iloc[0] grabs the station name from the index
            best_BF_method[station_name] = best_method

            #extract the baseflow timeseries for the best method
            obs_Qb = obs_bf_dict[best_method]
            obs_bfi = obs_bfi[best_method]

            #use the same best method to extract the baseflow from the simulated Q
            sim_bf_dict, sim_bfi = baseflow.separation(q_merged[['q_model']], return_bfi=True, return_kge=False, method=best_method)
            sim_Qb= sim_bf_dict[best_method]
            sim_bfi = sim_bfi[best_method]

            #combine into a dataframe
            bf_df = pd.concat([obs_Qb, sim_Qb], axis=1)
            bf_df.columns = ['obs_Qb', 'sim_Qb']
            bf_df['station'] = station_name
            
            #rearrange columns
            bf_df=bf_df[["station", 'obs_Qb', 'sim_Qb']]
            stations_Qb.append(bf_df)

            #bfi data
            bfi_df=pd.DataFrame({"obs_bfi":obs_bfi.values, "sim_bfi":sim_bfi.values})
            bfi_df['name'] = station_name
            bfi_df['model'] = model

            bfi_df=bfi_df[["name","obs_bfi","sim_bfi", "model"]]
            stations_BFI.append(bfi_df)


        if stations_Qb:
            model_bf_df = pd.concat(stations_Qb)
            model_bf_df['model'] = model
            all_models_Qb[model] = model_bf_df
        
        if stations_BFI:
            model_bfi_df = pd.concat(stations_BFI)
            model_bfi_df['model'] = model
            # Optionally store BFI data if needed
            all_models_BFI[model] = model_bfi_df

    return all_models_Qb, all_models_BFI, best_BF_method
#==============================================================================================
def extract_scenario_multistation_baseflow(base_sim_dir, models, eval_Obs, best_bfi_method_dict):
    """  
    Extract baseflow time series for each station and model using the best method based on observed data.  
    Parameters  
    ----------
    base_sim_dir : str
        Base directory containing model subdirectories.
    models : list of str
        List of model names corresponding to subdirectory names.
    eval_Obs : dict  
        Dictionary of observed discharge time series (keyed by station name). 
    best_bfi_method_dict: dict
        Dictionary containing best baseflow extraction method per station, based on observed data.

    Returns  
    -------
    dict of pd.DataFrame  
        Dictionary with model names as keys and DataFrames with baseflow time series as values.
    """
    # Ensure station names are uppercase for consistency
    eval_Obs_upper = keys_upper(eval_Obs)

    all_models_Qb = {}
    all_models_BFI = {}

    for model in models:
        print(f'\nExtracting baseflow for {model}...')
        sim_dir = Path(base_sim_dir) / model / 'Qrouted'
        stations_Qb = []
        stations_BFI = []

        for fpath in sorted(sim_dir.glob("*.csv")):
            station_name = fpath.stem.upper()

            if station_name not in eval_Obs_upper:
                continue

            sim_Q = pd.read_csv(fpath, index_col=0, parse_dates=True)

            #select best method based on observed baseflow separation
            if station_name not in best_bfi_method_dict:
                continue

            bfi_method=best_bfi_method_dict[station_name]

            #extract baseflow
            msg = f"Extracting baseflow for {station_name}"
            sys.stdout.write("\r" + msg + "  " * 20)
            sys.stdout.flush()

            #use the same best method to extract the baseflow from the simulated Q
            sim_bf_dict, sim_bfi = baseflow.separation(sim_Q, return_bfi=True, return_kge=False, method=bfi_method)
            sim_Qb= sim_bf_dict[bfi_method]
            sim_bfi = sim_bfi[bfi_method]

            sim_Qb.columns = ['sim_Qb']
            sim_Qb['station'] = station_name
            
            #rearrange columns
            sim_Qb=sim_Qb[["station", 'sim_Qb']]
            stations_Qb.append(sim_Qb)

            #bfi data
            sim_bfi_df=pd.DataFrame({"sim_bfi":sim_bfi.values})
            sim_bfi_df['name'] = station_name
            sim_bfi_df['model'] = model

            sim_bfi_df=sim_bfi_df[["name","sim_bfi", "model"]]
            stations_BFI.append(sim_bfi_df)

        if stations_Qb:
            model_bf_df = pd.concat(stations_Qb)
            model_bf_df['model'] = model
            all_models_Qb[model] = model_bf_df
        
        if stations_BFI:
            model_bfi_df = pd.concat(stations_BFI)
            model_bfi_df['model'] = model
            # Optionally store BFI data if needed
            all_models_BFI[model] = model_bfi_df

    return all_models_Qb, all_models_BFI
#==============================================================================================
def seasonal_baseflow_analysis(all_models_Qb, models, season_map, Q_names):
    """  
    Analyze seasonal mean baseflow for each station and model.  
    Parameters  
    ----------
    all_models_Qb : dict of pd.DataFrame  of models and station baseflow
        Dictionary with model names as keys and DataFrames with baseflow time series as values.  
    models : list of str  
        List of model names corresponding to keys in all_models_Qb.  
    season_map : dict
        Mapping of month numbers to season labels.  
    Q_names: list  
        Name of discharge to groupby e.g. ["obs_Qb","sim_Qb"]

    Returns  
    -------
    pd.DataFrame  
        DataFrame with columns ['name', 'model', 'season', 'obs_Qb', 'sim_Qb'] containing seasonal mean baseflow.  
    """

    seasonal_baseflow = []

    # Initialize a dictionary to hold seasonal data for the station
    for model in models:
        model_df = all_models_Qb[model]

        stations_Qb = []  # <-- collect per model

        for nameStation in model_df['station'].unique():

            station_df = model_df[model_df['station'] == nameStation]
            
            station_df = station_df.copy()
            station_df["season"] = station_df.index.month.map(season_map)

            # seasonal climatology
            season_Qb = station_df.groupby("season")[Q_names].mean().reset_index()
            season_Qb["name"] = nameStation
            season_Qb["model"] = model
            stations_Qb.append(season_Qb)
        
        if stations_Qb:  # <-- use the correct list
            model_Qb_df = pd.concat(stations_Qb)
            seasonal_baseflow.append(model_Qb_df)
        
        #convert to dataframe
    seasonal_baseflow_df = pd.concat(seasonal_baseflow)
    #rearrange columns
    seasonal_baseflow_df = seasonal_baseflow_df[["name", "model", "season"] + Q_names]

    return seasonal_baseflow_df
#==================================================

def plot_multimodel_spread(flow_df: dict, seasons: dict,
                           obs_name: str, sim_name: str,
                           xlabel: str,
                           ylabel: str):
    """
    Plot the spread of simulated vs observed baseflow for multiple models with 95PPU
    and compute the P-factor and R².
    """
    plt.figure(figsize=(8,6), dpi=120)

    all_obs = []
    all_mean_sim = []
    for season in seasons:
        for station in flow_df['name'].unique():
            q_sim_season = flow_df.loc[
                (flow_df['season'] == season) &
                (flow_df['name'] == station), sim_name]
            
            eval_Obs_season = flow_df.loc[
                (flow_df['season'] == season) &
                (flow_df['name'] == station), obs_name]

            mean_sim = q_sim_season.mean()
            min_sim = q_sim_season.min()
            max_sim = q_sim_season.max()
            x_obs = eval_Obs_season.mean()

            all_obs.append(x_obs)
            all_mean_sim.append(mean_sim)

            plt.errorbar(
                x_obs, mean_sim,
                yerr=[[mean_sim - min_sim], [max_sim - mean_sim]],
                fmt='o', color='dodgerblue', markersize=5,
                alpha=0.6, capsize=2, ecolor='gray', elinewidth=0.7, capthick=0.6
            )

    # arrays
    all_obs = np.array(all_obs)
    all_mean_sim = np.array(all_mean_sim)

    # ---- Compute R² ----
    r2 = r2_score(all_obs, all_mean_sim)
    print(f"Coefficient of determination (R²): {r2:.2f}")

    # ---- Compute r ----
    r = np.corrcoef(all_obs, all_mean_sim)[0, 1]
    print(f"Correlation coefficient (r): {r:.2f}")

    # residuals and 95PPU
    residuals = all_mean_sim - all_obs
    lower = np.percentile(residuals, 2.5)
    upper = np.percentile(residuals, 97.5)

    # 1:1 line
    lims = [0, 60]
    plt.plot(lims, lims, 'r-', lw=1)

    # 95PPU band
    x_line = np.linspace(lims[0], lims[1], 200)
    plt.fill_between(x_line, x_line + lower, x_line + upper,
                     color='gray', alpha=0.2, label='95PPU')

    # ---- Compute P-factor ----
    inside = ((all_mean_sim >= all_obs + lower) & (all_mean_sim <= all_obs + upper))
    p_factor = inside.mean() * 100
    print(f"P-factor (percentage of points within 95PPU): {p_factor:.1f}%")

    # annotate R² and P-factor on plot
    plt.text(0.05, 0.92,
             f"$r$ = {r:.2f}",
             transform=plt.gca().transAxes,
             fontsize=11, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # axes, labels
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.grid(alpha=0.4)
    plt.show()
#========================================================================
#Calculate multimodel ensemble BFI mean
#Calculate multimodel ensemble BFI mean
def ensemble_station_BFI(all_models_BFI, sim_name):
    """  
    Concatenate simulated BFI from multiple model dictionaries
    ----------
    Parameters:
    all_model_BFI: dict  
    Dictionary containing individual model station dataframes of BFI, indexed by model name
    smi_name: str
    name of simulate Q variable

    ----------
    Returns:  
    df: ensemble average dataframe of average per station BFI
    """
    # Combine all model BFIs by 'name'
    ensemble_bfi = pd.concat(
        [df.set_index('name')[sim_name] for df in all_models_BFI.values()],
        axis=1
    )

    # Compute mean across models
    ensemble_bfi_mean = ensemble_bfi.mean(axis=1, skipna=True)

    # Convert to final DataFrame
    ensemble_bfi_df = ensemble_bfi_mean.reset_index()
    ensemble_bfi_df.columns = ['name', 'ens_BFI']

    return ensemble_bfi_df
#======================================================================
def df_to_gdf(df, stations_df):
    """ Convert dataframe to geodataframe by combining with df with station coordinates"""
    #change station_locations to upper
    stations_df['name'] = stations_df['name'].str.upper()

    #only unique station names
    station_locations_df = stations_df.drop_duplicates(subset='name').set_index('name')

    #merge station_locations with df on 'name'
    ensemble_stations = pd.concat([station_locations_df, df], axis=1).dropna()

    #to geodataframe
    gdf_ = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(ensemble_stations['lon'], ensemble_stations['lat']))
    gdf_.crs = "EPSG:4326"

    return gdf_

#========================================================================
def ensemble_BFI_to_geodataframe(ensemble_BFI_df, obs_BFI_df, stations_df):
        """  
        Merge ensemble of station geodataframes with station files to get a gdf to map

        -----------
        Parameters:
            ensemble_df:
            df with ensemble BFI per station
            observed_bfi_df:
            df with observed bfi per station
            stations_df:
            df with station names and coordinates
            
        -------
        Returns:
        gdf:
        geodataframe with station names, locations and obs and sim ems. BFI
        
        """
        #merge with observed BFI
        merged_BFI = pd.concat([ensemble_BFI_df.set_index('name'), obs_BFI_df[['name','obs_bfi']].set_index('name')], axis=1)
        merged_BFI['bfi_diff'] = merged_BFI['ens_BFI'] - merged_BFI['obs_bfi']

        #Merge with stations and convert to gdf
        bfi_diff_gdf = df_to_gdf(merged_BFI, stations_df)

        return bfi_diff_gdf
#===========================================================================================
def wilcoxon_stats(ens_BFI):
# obs_bfi and sim_bfi are 1D arrays aligned by station
    obs_bfi = ens_BFI['obs_bfi']
    sim_bfi = ens_BFI['ens_BFI']

    assert len(obs_bfi) == len(sim_bfi)
    n = len(obs_bfi)
    diff = sim_bfi - obs_bfi

    # common bins
    bins = np.linspace(min(obs_bfi.min(), sim_bfi.min()),
                    max(obs_bfi.max(), sim_bfi.max()), 25)

    # bootstrap CI helper
    def mean_ci(a, nboot=10000, alpha=0.05, rng=None):
        rng = np.random.default_rng(rng)
        boot = rng.choice(a, (nboot, a.size), replace=True).mean(axis=1)
        return a.mean(), np.quantile(boot, [alpha/2, 1-alpha/2])

    m_obs, (lo_obs, hi_obs) = mean_ci(obs_bfi)
    m_sim, (lo_sim, hi_sim) = mean_ci(sim_bfi)
    m_diff, (lo_diff, hi_diff) = mean_ci(diff)

    # paired test (choose one)
    tstat, p_t = stats.ttest_rel(sim_bfi, obs_bfi)      # parametric
    wstat, p_w = stats.wilcoxon(sim_bfi, obs_bfi)       # nonparametric

    return m_diff, p_w

#===========================================================================================
#### Map the BFI and KGE values
def map_BFI_anomalies(bfi_gdf, boundaries_shp, streams_shp, m_diff, p_w):

    """ 
    Map differences between obs. and sim. BFI per station overlain over a map and river network
    ------
    Parameters:
    bfi_gdf:
    Geodataframe containing obs-sim BFI, station names and coordinates
    boundaries_shp:  
    Boundary of domain
    streams_shp:
    river network
    m_diff, p_w: 
    wilcoxon stats above

    -----
    Returns:
    map of sim-obs. BFI per station and histogram of error distribution
    
    """

    fig, ax = plt.subplots(figsize=(15, 7), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)

    boundaries_shp.plot(ax=ax, linewidth=0.5,
                edgecolor='black', facecolor='none', zorder=1, transform=ccrs.PlateCarree())

    streams_shp.plot(ax=ax, linewidth=0.5, alpha=0.2,
                    edgecolor='dodgerblue', facecolor='none', zorder=2, transform=ccrs.PlateCarree())


    diff = bfi_gdf['bfi_diff'].to_numpy()

    # color: diverging, centered at 0
    vmax = np.nanpercentile(np.abs(diff), 99)  # robust limits
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = 'coolwarm'  # blue = negative, red = positive (reverse if you prefer)

    # size: scale |diff| to [smin, smax]
    mag = np.abs(diff)
    smin, smax = 30, 300
    sizes = np.interp(mag, [mag.min(), mag.max()], [smin, smax])


    sc = ax.scatter(
        bfi_gdf.geometry.x, bfi_gdf.geometry.y,
        c=diff,
        s=50,
        cmap=cmap,
        edgecolor='white',
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    gl = ax.gridlines(draw_labels=True, color='gray', lw=0.6, alpha=0.2)
    gl.xlocator = plt.FixedLocator(np.arange(0, 10, 1))
    gl.ylocator = plt.FixedLocator(np.arange(49.5, 51.9, 0.4))
    gl.top_labels = True
    gl.right_labels = False

    #Insert inset histogram of BFI values
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_in = inset_axes(ax, width="30%", height="25%", bbox_to_anchor=(-.60, -.62, 1.0, 1.0), bbox_transform=ax.transAxes)
    ax_in.axvline(0, linewidth=1, color='red')
    ax_in.axvline(m_diff, linestyle='--')
    #ax_in.axvspan(lo_diff, hi_diff, alpha=0.2)
    ax_in.text(0.6, 0.95, f'mean={m_diff:.02f},  \np={p_w:.3f}',
            transform=ax_in.transAxes, va='top', fontsize=7)
    cm = mpl.colormaps[cmap]
    _, bins, patches = ax_in.hist(diff, bins=20, color="r")  # Corrected axis
    bin_centers = 0.5*(bins[:-1]+bins[1:])
    col = bin_centers - min(bin_centers)
    if np.max(col) > 0:
        col /= np.max(col)

    for c, p in zip(col, patches):
        plt.setp(p, "facecolor", cm(c))
        edgecolor = 'gray'
        lwidth = 0.4
        plt.setp(p, "edgecolor", edgecolor, "linewidth", lwidth)
    ax_in.set_xlabel(r'$BFI_{sim} - BFI_{obs}$', fontsize=9)
    ax_in.set_ylabel('Stations', fontsize=9)
    ax_in.tick_params(labelsize=9)

    cax = fig.add_axes([0.291, 0.02, 0.44, 0.04]) #left, bottom, width, height
    cmap = plt.get_cmap(cmap)

    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('BFI Difference', fontsize=10, weight='bold')