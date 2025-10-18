
import os
import sys
import pandas as pd
from dask.diagnostics import ProgressBar
from dask import delayed, compute
import numpy as np
import shutil
import openpyxl
import xarray as xr
import glob
import pickle
import matplotlib.pyplot as plt
import zipfile
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#//////////////////////////////////////

def unzip_files(zipped_folder, unzipped_folder):
    """
    Unzips all .zip files in the specified folder to a new folder.
    Parameters:
    zipped_folder (str): Path to the folder containing .zip files.
    unzipped_folder (str): Path to the folder where files will be extracted.

    Returns:
    Unzipped files in the specified unzipped dir.
    """
    os.makedirs(unzipped_folder, exist_ok=True)

    zip_files = [f for f in os.listdir(zipped_folder) if f.lower().endswith('.zip')]

    for i, zipfile_name in enumerate(zip_files, 1):
        zip_path = os.path.join(zipped_folder, zipfile_name)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                print(f"[{i}/{len(zip_files)}] Extracting {zipfile_name}...", end ='\r')
                zf.extractall(path=unzipped_folder)
        except zipfile.BadZipFile:
            print(f"Skipping corrupt file: {zipfile_name}")

#============================================================================================================================
def copy_to_common_folder(src_folder, common_folder):
    """
    Copies all files from subdirectories within the source folder to the common folder.

    Parameters:
    src_folder (str): Path to the folder containing unzipped subfolders.
    common_folder (str): Path to the common folder where files will be copied.
    """
    # Ensure the common folder exists
    os.makedirs(common_folder, exist_ok=True)

    # Iterate through each subfolder in the source folder
    for subfolder in os.listdir(src_folder):
        subfolder_path = os.path.join(src_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.xlsx') or file_name.endswith('.csv'):
                    src_file = os.path.join(subfolder_path, file_name)
                    dst_file = os.path.join(common_folder, file_name)

                    shutil.copy(src_file, dst_file)
                    print(f"Copying {file_name} from {subfolder} to {common_folder}...", end='\r')

#============================================================================================================================

def extract_timeseries_wallonie(source_folder):
    """
    Extracts time series data and metadata from Hydrométrie Wallonie Excel files.

    Parameters:
    source_folder (str): Folder containing Wallonie .xlsx files.

    Returns:
    station_Q (dict): Dictionary of station names → time series DataFrames.
    info_df (pd.DataFrame): DataFrame with station names as index and lat/lon as columns.
    """
    station_info = {}  # Metadata per station
    station_Q = {}     # Time series per station

    files = glob.glob(os.path.join(source_folder, '*.xlsx'))

    for f in files:
        try:
            df = pd.read_excel(f, engine='openpyxl')
            filename = os.path.basename(f)
            
            # Extract header info
            df_header = df.head(8)
            station_name = str(df_header.iloc[0, 1]).strip()
            lat = float(df_header.iloc[1, 1])
            lon = float(df_header.iloc[2, 1])
            station_info[station_name] = {'lat': lat, 'lon': lon}

            # Extract time series
            data = df.iloc[9:, [0, 1]].copy()
            data.columns = ['Date', 'Q']
            data.dropna(subset=['Date'], inplace=True)
            data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
            data.set_index('Date', inplace=True)
            data = data.resample('D').mean()
            data.sort_index(inplace=True)

            station_Q[station_name] = data
        except Exception as e:
            print(f"Failed to process {f}: {e}")

    info_df = pd.DataFrame.from_dict(station_info, orient='index')

    info_df = info_df.reset_index()
    #rename the columns
    info_df.columns = ['station_name', 'station_latitude', 'station_longitude']

    return station_Q, info_df

#============================================================================================================================

def load_or_extract_wallonie_data(dict_path, df_path, compute_func, *args, **kwargs):
    """
    Load Q_dict and station_coords from pickle if available,
    otherwise run compute_func and cache the results.
    Args:
        dict_path (str): Path to the pickle file for Q_dict.
        df_path (str): Path to the pickle file for station_coords.
        compute_func (callable): Function to compute Q_dict and station_coords.
        *args: Positional arguments for compute_func.
        **kwargs: Keyword arguments for compute_func.
    Returns:
        tuple: Q_dict and station_coords.

    """
    if os.path.exists(dict_path) and os.path.exists(df_path):
        print("Loading cached Wallonie discharge data...", end='\r')
        with open(dict_path, 'rb') as f1, open(df_path, 'rb') as f2:
            return pickle.load(f1), pickle.load(f2)
    else:
        print("Processing Wallonie discharge data (first time)...", end='\r')
        Q_dict, station_coords = compute_func(*args, **kwargs)
        with open(dict_path, 'wb') as f1:
            pickle.dump(Q_dict, f1)
        with open(df_path, 'wb') as f2:
            pickle.dump(station_coords, f2)
        return Q_dict, station_coords
    print("Data processing complete!")


#============================================================================================================================
#Extract subset of stations for model evaluation based on peak discharge
def extract_eval_stations(Q_dict, threshold_max, min_length_days):
    """
    Extracts stations for model evaluation based on peak discharge and non-NaN data length.
    
    Parameters:
    Q_dict (dict): Dictionary of station names → time series DataFrames with a 'Q' column.
    threshold_max (float): Minimum peak discharge (Q) required for inclusion.
    min_length_days (int): Minimum number of valid (non-NaN) daily discharge values required.
    
    Returns:
    eval_stations (dict): Dictionary of stations that meet the criteria.
    """
    eval_stations = {}

    for station_name, df in Q_dict.items():
        if 'Q' not in df.columns:
            continue  # Skip if 'Q' column is missing

        q_valid = df['Q'].dropna()
        max_Q = q_valid.max()
        valid_days = len(q_valid)

        if max_Q > threshold_max and valid_days >= min_length_days:
            eval_stations[station_name] = df
        # Optionally log excluded stations:
        # else:
        #     print(f"Excluded {station_name}: max Q = {max_Q}, valid days = {valid_days}")

    return eval_stations

#============================================================================================================================
def extract_timeseries_from_netCDF(
    dataset_path, 
    station_coordinates_csv, 
    var, 
    outDir, 
    num_workers=8
):
    """
    Extract time series from a NetCDF file for multiple station coordinates using Dask,
    and save each station's time series as a separate CSV file.

    Parameters
    ----------
    dataset_path : str
        Path to the NetCDF file.
    station_coordinates_csv : str
        Path to the CSV file with columns: name, lat, lon.
    var : str
        Variable to extract from the NetCDF file.
    outDir : str
        Base output directory where CSVs will be saved.
    num_workers : int, optional
        Number of parallel threads to use for writing files (default is 8).
    """
    
    # 1) Load station data and drop duplicates
    stations = pd.read_csv(station_coordinates_csv, encoding='utf-8')

    stations = stations.drop_duplicates(subset='name', keep='last')

    names = stations['name'].values
    lats = stations['lat'].values
    lons = stations['lon'].values

    # 2) Load dataset with chunking
    ds = xr.open_dataset(dataset_path, chunks={'time': 500})

    if var not in ds:
        raise ValueError(f"Variable '{var}' not found in the dataset.")

    # 3) Select variable at all station locations
    point_da = ds[var].sel(
        lat=xr.DataArray(lats, dims='station'),
        lon=xr.DataArray(lons, dims='station'),
        method='nearest'
    )

    # 4) Convert to wide DataFrame
    df = point_da.to_dataframe(name=var).unstack('station')[var]
    df.columns = names

    # 5) Create output directory
    out_dir = os.path.join(outDir, var)
    os.makedirs(out_dir, exist_ok=True)

    # 6) Define per-station write function
    def write_station(name, series):
        out_path = os.path.join(out_dir, f'{name}.csv')
        if isinstance(series, pd.DataFrame):
            if series.shape[1] > 1:
                print(f"Skipping '{name}' — multiple columns detected")
                return None
            series.columns = [var]
        else:
            series = series.to_frame(name=var)
        series.to_csv(out_path, index=True)
        return out_path

    # 7) Prepare parallel write tasks
    written_names = set()
    tasks = []
    for name in names:
        if name in written_names or name not in df.columns:
            continue
        written_names.add(name)
        tasks.append(delayed(write_station)(name, df[name]))

    # 8) Compute with Dask thread pool
    with ProgressBar():
        results = compute(*tasks, scheduler='threads', num_workers=num_workers)

    print(f"\n {len([r for r in results if r])} files written to: {out_dir}")

#=============================================================================================================================
#MODEL PERFORMANCE

def compute_model_metrics(observed, simulated, epsilon=1e-6):
    """
    Computes NSE, KGE, PBIAS, and LNSE between observed and simulated data.

    Parameters:
    - observed: array-like of observed values
    - simulated: array-like of simulated values
    - epsilon: small constant to avoid log(0) in LNSE

    Returns:
    - metrics (dict): Dictionary with rounded values of all metrics
    """
    observed = np.array(observed)
    simulated = np.array(simulated)

    # Drop NaNs and align

    if len(observed) == 0 or np.std(observed) == 0:
        return {'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan, 'LNSE': np.nan}

    # NSE
    nse_denom = np.sum((observed - np.mean(observed)) ** 2)
    nse = 1 - (np.sum((observed - simulated) ** 2) / nse_denom) if nse_denom != 0 else np.nan

    # KGE
    if np.std(simulated) == 0 or np.mean(observed) == 0:
        kge = np.nan
    else:
        # Convert and flatten arrays
        x = np.array(observed, dtype=float).flatten()
        y = np.array(simulated, dtype=float).flatten()

        # Drop NaNs from both
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        # Now safely calculate correlation
        if len(x) >= 2:
            r = np.corrcoef(x, y)[0, 1]
        else:
            r = np.nan  # Not enough data

        # Calculate KGE components
        if np.std(simulated) == 0 or np.mean(observed) == 0:
            kge = np.nan
        else:
            beta = np.mean(simulated) / np.mean(observed)
            gamma = np.std(simulated) / np.std(observed)
            kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

    # PBIAS
    pbias = 100 * np.sum(simulated - observed) / np.sum(observed) if np.sum(observed) != 0 else np.nan

    # LNSE
    # Make sure you're working with NumPy arrays of float type
    observed = np.array(observed, dtype=float)
    simulated = np.array(simulated, dtype=float)

    # Drop any NaNs (or infinite values if needed)
    mask = ~np.isnan(observed) & ~np.isnan(simulated)
    observed = observed[mask]
    simulated = simulated[mask]

    # Now apply log safely
    epsilon = 1e-10
    log_obs = np.log(observed + epsilon)
    log_sim = np.log(simulated + epsilon)
    lnse_denom = np.sum((log_obs - np.mean(log_obs)) ** 2)
    lnse = 1 - (np.sum((log_obs - log_sim) ** 2) / lnse_denom) if lnse_denom != 0 else np.nan


    return {
        'NSE': np.round(nse, 2),
        'KGE': np.round(kge, 2),
        'PBIAS': np.round(pbias, 2),
        'LNSE': np.round(lnse, 2)
    }
#=============================================================================================================================
#multistation model performance
def status(msg, width=80):
    """
    Print msg overwriting the previous line.
    Pads/truncates to `width` so old text is fully cleared.
    """
    sys.stdout.write('\r' + msg.ljust(width)[:width])
    sys.stdout.flush()


def compute_multistation_metrics(simulated_Q_dir, eval_stations_Q, cal_start, cal_end, val_start, val_end):
    """
    Computes model performance metrics for multiple stations.

    Parameters:
    - simulated_Q_dir (str): Directory containing simulated discharge CSV files.
    - eval_stations_Q (dict): Dictionary of observed discharge DataFrames.
    - cal_start (str): Start date for calibration period (YYYY-MM-DD).
    - cal_end (str): End date for calibration period (YYYY-MM-DD).
    - val_start (str): Start date for validation period (YYYY-MM-DD).
    - val_end (str): End date for validation period (YYYY-MM-DD).
    Returns:
    - model_metrics_cal (dict): Dictionary of calibration metrics for each station: NSE, KGE, PBIAS, LNSE.
    - model_metrics_val (dict): Dictionary of validation metrics for each station.
    """

    model_metrics_cal = {}
    model_metrics_val = {}

    sim_df_files = glob.glob(os.path.join(simulated_Q_dir, '*.csv'))

    #sort the files
    sim_df_files.sort()

    for file in sim_df_files:
        station = os.path.basename(file).split('.')[0]

        df_sim = pd.read_csv(file, parse_dates=['time'], index_col='time')

        if station not in eval_stations_Q:
            status(f"Station {station} not found, skipping to the next station.")
            continue

        df_obs = eval_stations_Q[station].copy()
        df_obs.replace(-9999, np.nan, inplace=True)
        df_obs = df_obs.resample('D').mean()
        df_obs.columns = ['Q']

        obs_sim = pd.concat([df_obs, df_sim], axis=1)
        obs_sim.columns = ['observed', 'simulated']
        obs_sim.dropna(inplace=True)

        # Calibration period
        cal = obs_sim[cal_start:cal_end]
        if len(cal) >= 1825:
            model_metrics_cal[station] = compute_model_metrics(cal['observed'], cal['simulated'])
        else:
            status(f"Calibration period too short for {station} ({len(cal)} days)")
            model_metrics_cal[station] = None

        # Validation period
        val = obs_sim[val_start:val_end]
        if len(val) >= 1825:
            model_metrics_val[station] = compute_model_metrics(val['observed'], val['simulated'])
        else:
            status(f"Validation period too short for {station} ({len(val)} days)")
            model_metrics_val[station] = None

    return model_metrics_cal, model_metrics_val



#=============================================================================================================================
#CONVERT DICTIONARY TO DATAFRAME
def dict_to_df(metrics_dict):
    """
    Convert a dictionary of metrics to a DataFrame.
    
    Parameters:
    metrics_dict (dict): Dictionary of metrics with station names as keys.
    
    Returns:
    pd.DataFrame: DataFrame containing the metrics.
    """
    df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'name'}, inplace=True)
    return df

#=============================================================================================================================
#TIMESERIES PLOTTING

def plot_station_timeseries(station_name, observed_dict,
                             sim_dir, cal_metrics_df,
                             val_metrics_df,
                             calPeriod_end):
    """
    Plot observed and simulated time series for a given station and display NSE values.

    Parameters:
    -----------
    station_name : str
        Name of the station.
    observed_dict : dict
        Dictionary of observed Series with datetime index.
    sim_dir : str
        Directory where simulated CSVs are stored.
    cal_metrics_df : pd.DataFrame
        Calibration metrics.
    val_metrics_df : pd.DataFrame
        Validation metrics.
    calPeriod_end : str
        Calibration end date (format: 'YYYY-MM-DD').
    """

    # Load observed
    if station_name not in observed_dict:
        print(f" Station '{station_name}' not found in observed data.")
        return
    obs = observed_dict[station_name].copy()
    obs = obs.replace(-9999, pd.NA).resample('D').mean()

    # Load simulated
    sim_file = os.path.join(sim_dir, f"{station_name}.csv")
    if not os.path.exists(sim_file):
        print(f"Simulation file not found for station: {sim_file}")
        return
    sim = pd.read_csv(sim_file, parse_dates=['time'], index_col='time')

    # Merge and clean
    df = pd.concat([obs, sim], axis=1, join='inner')
    df.columns = ['Observed', 'Simulated']
    df['Simulated'] = df['Simulated'].where(df['Observed'].notna(), np.nan)

    # Trim to overlapping date range
    start_date = max(df['Observed'].dropna().index.min(), df['Simulated'].dropna().index.min())
    end_date = min(df['Observed'].dropna().index.max(), df['Simulated'].dropna().index.max())
    df = df[(df.index >= start_date) & (df.index <= end_date)].sort_index()

    # Extract NSE values if present
    nse_val = None
    nse_cal = None

    if station_name in val_metrics_df['name'].values:
        nse_val_row = val_metrics_df.loc[val_metrics_df['name'] == station_name, 'KGE']
        nse_val = nse_val_row.values[0] if not nse_val_row.empty else None

    if station_name in cal_metrics_df['name'].values:
        nse_cal_row = cal_metrics_df.loc[cal_metrics_df['name'] == station_name, 'KGE']
        nse_cal = nse_cal_row.values[0] if not nse_cal_row.empty else None

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=110)
    ax.plot(df.index, df['Observed'], label='Observed', color='k', linewidth=0.9)
    ax.plot(df.index, df['Simulated'], label='Simulated', color='dodgerblue', linewidth=0.8)

    # Display NSEs and optional vertical line
    cal_end_date = pd.to_datetime(calPeriod_end)

    if nse_cal is not None and nse_val is not None:
        ax.axvline(x=cal_end_date, color='red', lw=0.8)
        ax.text(0.02, 0.85, f'KGE val: {nse_val:.2f}', transform=ax.transAxes, fontsize=12)
        ax.text(0.75, 0.85, f'KGE cal: {nse_cal:.2f}', transform=ax.transAxes, fontsize=12)
    elif nse_cal is not None:
        ax.text(0.8, 0.85, f'KGE cal: {nse_cal:.2f}', transform=ax.transAxes, fontsize=12)
    elif nse_val is not None:
        ax.text(0.8, 0.85, f'KGE val: {nse_val:.2f}', transform=ax.transAxes, fontsize=12)

    ax.set_ylabel("Discharge (m$^3$/s)")
    ax.set_title(f"Discharge at {station_name}")
    ax.grid(True, alpha=0.5)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.show()

#=============================================================================================================================
#MAP MODEL PERFORMANCE

def map_model_stats(boundary_shp_path, rivers_shp_path,
                        calib_gdf, val_gdf,
                        performance_statistic, cmap='viridis'):
    """
    Maps calibration and validation model statistics side by side.

    Parameters
    ----------
    boundary_shp_path : str
        Path to the shapefile for the boundary.
    rivers_shp_path : str
        Path to the shapefile for rivers.
    calib_gdf : GeoDataFrame
        GeoDataFrame with calibration statistics.
    val_gdf : GeoDataFrame
        GeoDataFrame with validation statistics.
    performance_statistic : str
        Name of the performance column to plot (e.g., 'KGE', 'NSE').
    cmap : str or matplotlib Colormap
        Colormap to use for both plots.

    Returns
    -------
    None: Displays the plot.
    """
    # Read shapefiles
    boundary = gpd.read_file(boundary_shp_path)
    rivers = gpd.read_file(rivers_shp_path)

    # Filter positive values only
    calib_gdf = calib_gdf[calib_gdf[performance_statistic] > 0].copy()
    val_gdf = val_gdf[val_gdf[performance_statistic] > 0].copy()

    # Shared color normalization across both plots
    stat_min = min(calib_gdf[performance_statistic].min(), val_gdf[performance_statistic].min())
    stat_max = max(calib_gdf[performance_statistic].max(), val_gdf[performance_statistic].max())
    norm = Normalize(vmin=stat_min, vmax=stat_max)
    cmap = plt.get_cmap(cmap)

    # Marker size logic
    def compute_marker_sizes(gdf):
        if stat_max != stat_min:
            norm_stat = (gdf[performance_statistic] - stat_min) / (stat_max - stat_min)
            return norm_stat * 180
        else:
            return np.full(len(gdf), 40)

    calib_sizes = compute_marker_sizes(calib_gdf)
    val_sizes = compute_marker_sizes(val_gdf)

    # Set up side-by-side maps
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=110)
    titles = ['Calibration', 'Validation']
    datasets = [(calib_gdf, calib_sizes), (val_gdf, val_sizes)]

    for i, (gdf, sizes) in enumerate(datasets):
        ax = axes[i]

        # Plot boundary and rivers
        boundary.boundary.plot(ax=ax, edgecolor='gray', linewidth=1.0, alpha=0.5, transform=ccrs.PlateCarree())
        rivers.plot(ax=ax, edgecolor='dodgerblue', linewidth=0.5, alpha=0.3, transform=ccrs.PlateCarree())

        # Plot scatter of performance
        sc = ax.scatter(
            gdf.geometry.x, gdf.geometry.y,
            c=gdf[performance_statistic],
            s=sizes,
            cmap=cmap,
            norm=norm,
            edgecolor='black',
            linewidth=0.3,
            transform=ccrs.PlateCarree()
        )

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, color='gray', lw=0.6, alpha=0.2)
        gl.top_labels = gl.right_labels = False

        ax.set_title(f'{titles[i]}: {performance_statistic}', fontsize=14)

    # Shared colorbar
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='vertical', pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(performance_statistic, fontsize=12)

    plt.tight_layout()
    plt.show()

#=============================================================================================================================
def map_calib_val_stats(boundary_shp_path, rivers_shp_path,
                        calib_gdf, val_gdf,
                        performance_statistic, cmap='viridis'):
    """
    Maps calibration and validation model statistics side by side.
    """
    # Read shapefiles
    boundary = gpd.read_file(boundary_shp_path)
    rivers = gpd.read_file(rivers_shp_path)

    # Filter positive values only
    calib_gdf = calib_gdf[calib_gdf[performance_statistic] > 0].copy()
    val_gdf = val_gdf[val_gdf[performance_statistic] > 0].copy()

    # Shared color normalization across both plots
    stat_min = min(calib_gdf[performance_statistic].min(), val_gdf[performance_statistic].min())
    stat_max = max(calib_gdf[performance_statistic].max(), val_gdf[performance_statistic].max())
    norm = Normalize(vmin=stat_min, vmax=stat_max)
    cmap = plt.get_cmap(cmap)

    def compute_marker_sizes(gdf):
        if stat_max != stat_min:
            norm_stat = (gdf[performance_statistic] - stat_min) / (stat_max - stat_min)
            return norm_stat * 100
        else:
            return np.full(len(gdf), 40)

    calib_sizes = compute_marker_sizes(calib_gdf)
    val_sizes = compute_marker_sizes(val_gdf)

    # Create side-by-side subplots with a dedicated colorbar axis
    fig = plt.figure(figsize=(14, 7), dpi=180)

    # Define the axes for the two plots and the colorbar
    #plt.subplot2grid((nrows, ncols), (start_row, start_col), colspan=..., rowspan=...)
    ax1 = plt.subplot2grid((6, 25), (0, 0), rowspan=5, colspan=9, projection=ccrs.PlateCarree()) #ax1 occupies the first 9 columns
    ax2 = plt.subplot2grid((6, 25), (0, 9), rowspan=5, colspan=9, projection=ccrs.PlateCarree()) #ax2 occupies the next 9 columns
    
    # Colorbar axis - placed on the last row
    # After creating fig
    cax = fig.add_axes([0.04, 0.22, 0.68, 0.05])  # [left, bottom, width, height] in figure coords (0 to 1)


    #space between the two plots
    plt.subplots_adjust(wspace=0.5)

    axes = [ax1, ax2]
    titles = ['Calibration', 'Validation']
    datasets = [(calib_gdf, calib_sizes), (val_gdf, val_sizes)]

    for i, (gdf, sizes) in enumerate(datasets):
        ax = axes[i]

        boundary.boundary.plot(ax=ax, edgecolor='gray', linewidth=1.0, alpha=0.7, transform=ccrs.PlateCarree())
        rivers.plot(ax=ax, edgecolor='dodgerblue', linewidth=0.5, alpha=0.3, transform=ccrs.PlateCarree())

        sc = ax.scatter(
            gdf.geometry.x, gdf.geometry.y,
            c=gdf[performance_statistic],
            s=sizes,
            cmap=cmap,
            norm=norm,
            edgecolor='black',
            linewidth=0.3,
            transform=ccrs.PlateCarree()
        )

        # Gridlines and label control
        gl = ax.gridlines(draw_labels=True, color='gray', lw=0.6, alpha=0.2)
        #specify gridlines for both axes
        gl.xlocator = plt.FixedLocator(np.arange(0, 10, 1))
        gl.ylocator = plt.FixedLocator(np.arange(49.5, 51.9, 0.4))
        gl.top_labels = True
        gl.left_labels = True if i == 0 else False
        gl.right_labels = False
        gl.bottom_labels = False

        ax.set_title(f'{titles[i]}', fontsize=14)

    # Shared colorbar
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(performance_statistic, fontsize=12, weight='bold')

    plt.tight_layout()
    plt.show()
    return fig
#=============================================================================================================================

def map_calib_val_stats(boundary_shp_path, rivers_shp_path,
                        calib_gdf, val_gdf,
                        performance_statistic, cmap='Blues'):
    """
    Maps calibration and validation model statistics side by side with inset histograms.
    """
    # Read shapefiles
    boundary = gpd.read_file(boundary_shp_path)
    rivers = gpd.read_file(rivers_shp_path)

    # Filter positive values only
    calib_gdf = calib_gdf[calib_gdf[performance_statistic] > 0].copy()
    val_gdf = val_gdf[val_gdf[performance_statistic] > 0].copy()

    # Shared normalization
    stat_min = min(calib_gdf[performance_statistic].min(), val_gdf[performance_statistic].min())
    stat_max = max(calib_gdf[performance_statistic].max(), val_gdf[performance_statistic].max())
    norm = Normalize(vmin=stat_min, vmax=stat_max)
    cmap = plt.get_cmap(cmap)

    def compute_marker_sizes(gdf):
        if stat_max != stat_min:
            norm_stat = (gdf[performance_statistic] - stat_min) / (stat_max - stat_min)
            return norm_stat * 100
        else:
            return np.full(len(gdf), 40)

    calib_sizes = compute_marker_sizes(calib_gdf)
    val_sizes = compute_marker_sizes(val_gdf)

    # Create subplots
    fig = plt.figure(figsize=(14, 7), dpi=180)
    ax1 = plt.subplot2grid((6, 25), (0, 0), rowspan=5, colspan=9, projection=ccrs.PlateCarree())
    ax2 = plt.subplot2grid((6, 25), (0, 9), rowspan=5, colspan=9, projection=ccrs.PlateCarree())
    cax = fig.add_axes([0.04, 0.22, 0.68, 0.05])  # colorbar

    plt.subplots_adjust(wspace=0.5)

    axes = [ax1, ax2]
    titles = ['Calibration', 'Validation']
    datasets = [(calib_gdf, calib_sizes), (val_gdf, val_sizes)]

    for i, (gdf, sizes) in enumerate(datasets):
        ax = axes[i]

        boundary.boundary.plot(ax=ax, edgecolor='gray', linewidth=1.0, alpha=0.7, transform=ccrs.PlateCarree())
        rivers.plot(ax=ax, edgecolor='dodgerblue', linewidth=0.5, alpha=0.3, transform=ccrs.PlateCarree())

        sc = ax.scatter(
            gdf.geometry.x, gdf.geometry.y,
            c=gdf[performance_statistic],
            s=sizes,
            cmap=cmap,
            norm=norm,
            edgecolor='black',
            linewidth=0.3,
            transform=ccrs.PlateCarree()
        )

        gl = ax.gridlines(draw_labels=True, color='gray', lw=0.6, alpha=0.2)
        gl.xlocator = plt.FixedLocator(np.arange(0, 10, 1))
        gl.ylocator = plt.FixedLocator(np.arange(49.5, 51.9, 0.4))
        gl.top_labels = True
        gl.left_labels = True if i == 0 else False
        gl.right_labels = False
        gl.bottom_labels = False

        ax.set_title(f'{titles[i]}', fontsize=14)

        # Inset histogram  bbox_to_anchor = (x0, y0, width, height)
        inset_ax = inset_axes(ax, width="30%", height="25%", bbox_to_anchor=(-.58, -.62, 1.0, 1.0), bbox_transform=ax.transAxes)

        x = gdf[gdf[performance_statistic]>0.0][performance_statistic].to_numpy()

        cm = plt.cm.get_cmap(cmap)
        _, bins, patches = inset_ax.hist(x, bins=20, color="r")  # Corrected axis
        bin_centers = 0.5*(bins[:-1]+bins[1:])
        col = bin_centers - min(bin_centers)
        if np.max(col) > 0:
            col /= np.max(col)

        for c, p in zip(col, patches):
            plt.setp(p, "facecolor", cm(c))
            edgecolor = 'gray'
            lwidth = 0.4
            plt.setp(p, "edgecolor", edgecolor, "linewidth", lwidth)

        inset_ax.tick_params(axis='both', labelsize=10)
        inset_ax.set_xlabel(performance_statistic, fontsize=10)
        inset_ax.set_ylabel('No. of stations', fontsize=10)

        # Change frame color and width
        for spine in inset_ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_edgecolor('gray')


    # Shared colorbar
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(performance_statistic, fontsize=12, weight='bold')

    plt.tight_layout()

    #save the figure
    #plt.savefig("calibration_validation_map.png", bbox_inches='tight', dpi=300)
    return fig


