#Functions to post process mHM SMI outputs and related drought indices
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import mhm_drought_funcs as mdf
from joblib import Parallel, delayed
from itertools import product
import tqdm

#Segoe UI font
plt.rcParams['font.family'] = 'Segoe UI'
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def assign_date_ADM(drought_ADM, base_date):
    """
    Assigns start and end dates to the drought events in the results_ADM file based on mStart and mEnd.
    
    Parameters:
    drought_ADM (DataFrame): DataFrame containing drought periods with 'mStart' and 'mEnd' columns.
    base_date (datetime): The base date from which the months are calculated.
    mStart (int): The month number when the drought starts.
    mEnd (int): The month number when the drought ends.
    
    Returns:
    DataFrame: Updated DataFrame with 'start_month' and 'end_month' columns.
    """

    def add_months(months):
        return base_date + relativedelta(months=int(months)-1)

    drought_ADM['start_month'] = drought_ADM['mStart'].map(add_months)
    drought_ADM['end_month'] = drought_ADM['mEnd'].map(add_months)

    # Convert to datetime format
    drought_ADM['start_month'] = pd.to_datetime(drought_ADM['start_month'])
    drought_ADM['end_month'] = pd.to_datetime(drought_ADM['end_month'])

    return drought_ADM
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def ADM_lolliplot(drought_ADM, top_n_events):
    """
    Create a lollipop plot for drought events.
    
    Parameters:
    - drought_ADM: DataFrame containing drought event data.
    - top_n_events: Number of top drought events to highlight and label.
    
    Returns:
    - None: Displays the plot.
    """
    # Recalculate drought duration in months
    drought_ADM["duration"] = (drought_ADM["end_month"] - drought_ADM["start_month"]).dt.days // 30  # approximate months
    top_TDM = drought_ADM.nlargest(top_n_events, 'TDM')

    # Plot with Y-axis as duration and stems from start to end
    fig, ax = plt.subplots(figsize=(24, 9), dpi=200)
    sns.set_style("whitegrid")

    # Stems for all events representing duration
    for _, row in drought_ADM.iterrows():
        ax.plot([row['start_month'], row['start_month']], [0, row['TDM']],
                color='gray', linestyle='-', linewidth=0.2)

    # Hollow black circles for all events
    ax.scatter(drought_ADM['start_month'], drought_ADM['duration'],
            s=drought_ADM['TDM']/2, facecolors='gray', edgecolors='black', alpha=0.4, zorder=3, label='All Events')

    # Top 10: fiery filled circles sized by TDM
    sc = ax.scatter(top_TDM['start_month'], top_TDM['duration'],
                    s=top_TDM['TDM'] / 2, c=top_TDM['TDM'], cmap='YlOrRd',
                    edgecolor='black', alpha=0.9, zorder=5)

    # Colorbar for TDM
    cbar = plt.colorbar(sc, pad=0.01, aspect=20)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Total Drought Magnitude', fontsize=18)

    # Add labels adjacent to circles
    for _, row in top_TDM.iterrows():
        label = f"{row['start_month'].strftime('%b.%Y')}–{row['end_month'].strftime('%b.%Y')}"
        ax.text(row['start_month'], row['duration'] + 1.0, label, fontsize=18, ha='center', va='bottom', rotation=72)

    # Format axes
    ax.set_ylim(-1, drought_ADM['duration'].max() + 4)
    ax.set_ylabel("Drought Duration (months)", fontsize=18)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def cluster_percentiles(xr_data, xr_var, percentiles, labels):
    """
    Calculate specified percentiles for each time step in the xarray DataArray.
    
    Parameters:
    xr_data (xarray.DataArray): Input data array with time dimension.
    percentiles (list): List of percentiles to calculate.
    labels (list): List of labels corresponding to the percentiles.
    
    Returns:
    df (pandas.DataFrame): DataFrame containing the percentiles for each time step.
    """
    # Group by year and classify data
    results = []
    for year, data in xr_data.groupby('time'):
        
        # Classify data into percentile bins
        binned = xr.apply_ufunc(
            np.digitize,
            data[xr_var],
            input_core_dims=[[]],
            kwargs={'bins': percentiles},
        )
        
        # Total time steps in each bin:
        counts = [(binned == i).sum().item() for i in range(1, len(percentiles))]
        results.append([year] + counts)

    # Convert results into a DataFrame
    df = pd.DataFrame(results, columns=['Year'] + labels)
    df=df.set_index('Year')

    #calculate the percentage of each category
    df_perc = df.div(df.sum(axis=1), axis=0) * 100

    return df_perc
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def decadal_drought_area(smi_drought, decades, variable, drought_thresholds, labels):
    """
    Classify drought percentiles by decade and calculate area fractions.
    
    Parameters:
    - smi_drought: xarray DataArray with SMI data.
    - decades: List of decade strings in the format 'YYYY-YYYY'.
    - variable: Variable name for the SMI data.
    - drought_thresholds: List of thresholds for drought classification.
    - labels: List of labels for each drought category.
    
    Returns:
    - pd.DataFrame: DataFrame with area fractions for each drought category by decade.
    """
    drought_area_categories = {}

    #classify drought percentiles by decade
    for decade in decades:
        start_year, end_year = map(int, decade.split('-'))
        smi_decade = smi_drought.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))
        
        # Calculate the percentiles
        smi_decade_percentile_df = mdf.cluster_percentiles(smi_decade, variable, drought_thresholds, labels)
        drought_only = smi_decade_percentile_df.dropna(how='all') #drop rows where droughts are not present

        # Calculate the area fraction for each drought category
        area_fraction_per_category = drought_only.mean(axis=0)
        
        # Store the area fraction in the dictionary
        drought_area_categories[decade] = area_fraction_per_category
        
    # Convert the dictionary to a DataFrame
    drought_area_df = pd.DataFrame(drought_area_categories).T

    return drought_area_df

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def decadal_duration_category(smi_dataset, decades, drought_classes):
    """
    Count, per pixel, how many time steps fall in each drought class for each decade.

    Parameters
    ----------
    smi_dataset : xr.Dataset or xr.DataArray
        Must contain variable 'SMI' and a 'time' coordinate.
    decades : list[str]
        ['1971-1980', '1981-1990', ...]
    drought_classes : dict[str, tuple]
        {class_name: (low, high)} where low < SMI ≤ high.

    Returns
    -------
    dict[str, xr.Dataset]
        {decade: Dataset with one DataArray per drought class}
    """
    duration_dict = {}
    for decade in decades:
        start_year, end_year = map(int, decade.split('-'))
        smi_decade = smi_dataset.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))

        cls_durs = {}
        for cls, (lo, hi) in drought_classes.items(): 
            #lo, hi are the low and high SMI thresholds for each drought class
            # Create a mask for the drought class
            mask = (smi_decade['SMI'] > lo) & (smi_decade['SMI'] <= hi)

            # Count months (or time steps) in drought class per pixel
            dedade_dur = mask.sum(dim='time')

            # Propagate NaNs using the first timestep of the original SMI data
            #This ensures that we only count droughts in pixels where SMI is defined
            duration = dedade_dur.where(~smi_decade['SMI'].isel(time=0).isnull())

            #save the duration for each drought class in the dictionary
            cls_durs[cls] = duration
        
        # Convert the drought counts dict to an xarray Dataset
        cls_durs = xr.Dataset({name: duration for name, duration in cls_durs.items()})

        # store the drought counts for the decade in the duration_dict, with decade as the key
        duration_dict[decade] = cls_durs

    return duration_dict

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Drought categories per decade
def cluster_percentiles_pixelwise(smi_data, xr_var, drought_thresholds):
    """
    Classify each pixel's value into a percentile-based drought category.
    Parameters:
    - smi_data (xarray.Dataset): Input dataset containing the variable to classify.
    - xr_var (str): Name of the variable in the dataset to classify. (e.g., 'SMI').
    - drought_thresholds (list): List of thresholds defining the drought categories.

    Returns:
    - pandas.DataFrame: Mean percentage of each category per time step (area-aggregated).
    - xarray.DataArray: Label index (category) per time and pixel.
    """
    # Digitize values into bins defined by percentiles
    binned = xr.apply_ufunc(
        np.digitize,
        smi_data[xr_var],
        input_core_dims=[[]],
        kwargs={'bins': drought_thresholds, 'right': False},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[int]
    )
    # Keep bin indices for aggregation (do not map to strings yet)
    category_data = binned

    return category_data  # category_data contains integer bin IDs (1 = D4, 2 = D3, ...)


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def decadal_drought_clusters(decades, smi_data, bounds):
    """
    Classify drought percentiles by decade and calculate durations.
    Args:
        decades (list): List of decade strings in the format 'YYYY-YYYY'.
        smi_data (xarray.DataArray): Original SMI dataset with time dimension.
        bounds (list): List of percentile bounds for classification.
        
    Returns:
        decadal_durations (dict): Dictionary with average area percentages per category for each decade.
        decadal_categories (dict): Dictionary with per-decade maps of drought categories.
    """

    decadal_durations = {}
    decadal_categories = {}

    for decade in decades:
        start_year, end_year = map(int, decade.split('-'))
        smi_decade = smi_data.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))

        # Run pixelwise classification
        decadal_category_map = cluster_percentiles_pixelwise(smi_decade, 'SMI', bounds)
        
        # Replace category 5 with np.nan (assuming you only want D1-D4, where category indices are 1–4)
        decadal_category_map = decadal_category_map.where(decadal_category_map <= 4, np.nan)

        # Duration of drought for D1–D4 combined
        d1_to_d4_mask = (decadal_category_map >= 1) & (decadal_category_map <= 4)
        d1_to_d4_map = d1_to_d4_mask.sum(dim='time')

        # Mask out areas outside domain
        valid_mask = ~np.isnan(smi_data['SMI'][0])
        d1_to_d4_map = d1_to_d4_map.where(valid_mask, drop=False)
        
        # Store per-decade maps
        decadal_durations[decade] = {
            'D1_to_D4': d1_to_d4_map
        }

        # Also keep the full category map for advanced analysis
        decadal_categories[decade] = decadal_category_map

    return decadal_durations, decadal_categories

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
"""SPEI and SPI functions"""
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#apply a 90-day moving average to precipitation
def compute_SPI(pre, pre_var, timescale):
    """
    Compute the Standardized Precipitation Index (SPI) for a given timescale using the spei package of martinvonk
    https://github.com/martinvonk/SPEI
    
    Parameters:
    pre (xr.DataArray): Precipitation data.
    pre_var (str): The variable name in the DataArray that contains precipitation data.
    timescale (int): The timescale for the SPEI calculation in days. 30, 90, 120, 180 for SPI-1, SPI-3, SPI-4, SPI-6 respectively.
    Returns:
    xr.DataArray: The computed SPI values.
    """
    precip = pre[pre_var].rolling(time=timescale).sum() # Rolling sum to get the total precipitation over the timescale

    nlat, nlon = len(precip.lat), len(precip.lon)
    # Ensure the time dimension is at least as long as the timescale
    valid_time = precip.time[timescale-1:] #the first `timescale-1` entries will be NaN
    nvalid = len(valid_time)

    # Preallocate array for only valid SPI values
    spi_array = np.full((nlat, nlon, nvalid), np.nan)

    #the function is made to work with pd.Series, so we need to convert the xarray DataArray to a Series for each cell
    # and then compute the SPI for each cell
    def compute_spi_cell(j, i):
        tx_var = precip.isel(lat=j, lon=i).to_series()
        if np.isnan(tx_var).all() or np.std(tx_var) < 1e-6:
            return (j, i, np.full(nvalid, np.nan))
        spi = sp.spi(tx_var, dist=scs.gamma, fit_freq="ME") #the dist can be changed.
        # Remove the first timescale-1 NaNs
        spi = spi[timescale-1:]
        return (j, i, spi)

    results = Parallel(n_jobs=-1)(
        delayed(compute_spi_cell)(j, i)
        for j in tqdm(range(nlat), desc="Latitude rows")
        for i in range(nlon)
    )

    for j, i, spi_vals in results:
        spi_array[j, i, :] = spi_vals

    spi_da = xr.DataArray(
        spi_array,
        coords={
            'lat': precip.lat.values,
            'lon': precip.lon.values,
            'time': valid_time.values,
        },
        dims=['lat', 'lon', 'time'],
        name=f'spi_{timescale // 30}'  # Assuming timescale is in days, e.g., 30 for SPI-1
    )
    return spi_da
#--------------------------------------------------------------------------------


def compute_SPEI(p_excess, p_excess_var, timescale):
    """
    Compute the Standardized Precipitation-Evapotranspiration Index (SPEI) for a given timescale using the spei package of martinvonk

    Parameters:
    p_excess (xr.DataArray): Precipitation excess data (precipitation -
    evapotranspiration).
    timescale (int): The timescale for the SPEI calculation in days.
    Returns:
    xr.DataArray: The computed SPEI values.
    """

    nlat, nlon = len(p_excess.lat), len(p_excess.lon)
    ntime = len(p_excess.time)
    spei_array = np.full((nlat, nlon, ntime-(timescale-1)), np.nan)

    #The SPEI function requires a time series for each grid cell, so we will iterate over each cell
    # Function for one grid cell
    def compute_spei_cell(j, i):
        tx_var = p_excess[p_excess_var].isel(lat=j, lon=i).to_series()
        
        if np.isnan(tx_var).all() or np.std(tx_var) < 1e-6:
            #since the SPEI returns only non-nan values, effecttively after the first timescale-1 steps are skipped by the rolling sum.
            return (j, i, np.full(ntime-(timescale-1), np.nan)) 
        
        spei = sp.spei(tx_var, timescale=timescale, fit_freq="ME")
        return (j, i, spei)

    # Progress bar wrapper around the iterator
    results = Parallel(n_jobs=-1)(
        delayed(compute_spei_cell)(j, i)
        for j, i in tqdm(product(range(nlat), range(nlon)), total=nlat*nlon, desc="Grid cells")
    )

    # Fill result array
    for j, i, spei_vals in results:
        spei_array[j, i, :] = spei_vals

    #convert to xarray DataArray
    spei_da = xr.DataArray(
        spei_array,
        coords={
            'lat': p_excess.lat.values,
            'lon': p_excess.lon.values,
            'time': p_excess.time[timescale-1:]
        },
        dims=['lat', 'lon', 'time'],
        name='spei_1'
        )
    return spei_da
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

