import pandas as pd
import xarray as xr
from xarray import DataArray
import numpy as np
import geopandas as gpd
import rioxarray as rio
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import mapping
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


#function
def clip_to_region(shapefile, xr_dataset):
    """
    This function clips an xarray dataset to a given shapefile.

    Parameters
    ----------
    shapefile : geopandas.GeoDataFrame
        The shapefile to clip the dataset to.
    xr_dataset : xarray.Dataset
        The dataset to clip.

    Returns
    -------
    xarray.Dataset
        The clipped dataset.
    """
        #set shapefile to crs 4326
    shapefile = shapefile.to_crs('epsg:32631')

    #drop bnds dimension
    xr_dataset = xr_dataset.drop_dims("bnds", errors="ignore")

    #set spatial dimensions
    xr_dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

    #write crs
    xr_dataset.rio.write_crs('epsg:32631', inplace=True)

    #clip
    clipped = xr_dataset.rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True)

    return clipped

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def compute_optimal_h(data, bandwidths):
    """Function to find the optimal bandwidth for a kernel density estimate using cross-validated grid search.

    Parameters
    ----------
    data : numpy array
        Array of values to estimate the density function

    Returns
    -------
    optimal_h : float
        Optimal bandwidth for the kernel density estimate
    """
    # Define a range of bandwidths to test 
    # e.g. bandwidths = np.linspace(0.001, 0.9, 500)

    # Perform cross-validated grid search for bandwidth h
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=5)  # 5-fold cross-validation
    grid.fit(data[:, None])

    # Optimal bandwidth
    optimal_h = grid.best_params_['bandwidth']
    return optimal_h


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def compute_standardized_anomaly(ds: DataArray, freq) -> DataArray:
    """
    Compute the standardized anomaly of a DataArray.
    The standardized anomaly of each month with respect to all the corresponding months in the time series.
    For each month, the standardized anomaly is calculated as the anomaly divided by the standard deviation of the anomaly.
    Parameters
    ----------
    ds : DataArray
        The input rainfall data. At daily temporal scale.
    freq: string
        The frequency of target resampled data. 'M'/'ME' for monthly
    Returns
    -------
    DataArray
        The standardized anomaly.
    """
    # Step 1: Compute monthly total rainfall
    monthly_mean = ds.resample(time=freq).sum('time')

    # Step 2: Compute mean of each month across all years
    monthly_mean_grouped = monthly_mean.groupby('time.month').mean()

     # Step 3: Compute monthly anomalies
    # vectorized more efficient method
    ds_anomalies = monthly_mean.groupby('time.month') - monthly_mean_grouped

    # Step 4: Calculate the standard deviation of the anomalies for each month
    # Group anomalies by month and compute standard deviation over the time dimension
    anomalies_stdev_monthly = ds_anomalies.groupby('time.month').std()
    
    #compute the standardized monthly anomalies
    #Divide each monthly anomaly by the standard deviation of the corresponding month to get the standardized anomalies.
    standardized_anomalies = ds_anomalies.groupby('time.month') / anomalies_stdev_monthly

    return standardized_anomalies

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def group_weekly(dataset):
    """
    Groups an xarray dataset into weekly intervals, ensuring that:
    - The first week (Week 0) runs from January 1st to January 7th.
    - Each week consists of exactly 7 days, except for the final week, 
      which may be 1 or 2 days to include all remaining days in the year.

    Parameters:
    ----------
    dataset : xarray.Dataset
        An xarray dataset containing a time dimension.

    Returns:
    -------
    weekly_ds : xarray.Dataset
        A new dataset with weekly means computed, where:
        - The "time" coordinate represents the last day of each week.
        - All other dimensions (e.g., lat, lon) are preserved.
    """
    # Compute the custom week number
    weeks = ((dataset['time'].dt.dayofyear - 1) // 7)  # Subtract 1 to ensure 01/01 is within week 0

    # Initialize a list to store weekly data
    weekly_means = []

    for year in np.unique(dataset['time'].dt.year):  # Iterate over all years
        for i in np.unique(weeks.values):  # Iterate over unique weeks

            # Select data for the year
            datayear = dataset.sel(time=dataset['time'].dt.year == year)

            # Select data for the given week for the selected year
            dataweek = datayear.sel(time=(datayear['time'].dt.dayofyear - 1) // 7 == i)

            if dataweek['time'].size > 0:  # Ensure there is valid data
                startdate = dataweek['time'][0].values
                enddate = dataweek['time'][-1].values  # Last day of the selected week

                # Select data for the given time range and compute weekly mean
                weekly_data = dataset.sel(time=slice(startdate, enddate)).mean(dim='time')

                # Assign the weekly end date as the new time coordinate
                weekly_data = weekly_data.assign_coords(time=enddate, week=i)

                # Append the computed weekly mean to the list
                weekly_means.append(weekly_data)

    # Combine all weekly data into a single xarray dataset along the "time" dimension
    weekly_ds = xr.concat(weekly_means, dim="time")

    return weekly_ds

#==>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# For each var, extract the soil moisture content, field capacity, and wilting point at L1
def calculate_paw_smi(mhm_fluxes, soil_hydr_grid, sm_var, nhorizon):
    """
    Calculate the standardised plant available water (PAW) according to the European Drought Observatory (EDO)
    
    Parameters
    ----------
    mhm_fluxes : xarray.Dataset
        The mHM dataset containing the soil moisture content at L1
    
    soil_hydr_grid : xarray.Dataset
        The mHM dataset containing the soil hydraulic properties: field capacity and wilting point

    var : str
        The variable name of the soil moisture content at L1 (e.g. 'SWC_L01')
    nhorizon : int
        The horizon being considered (e.g. 0, 1, 2, ...)
        The horizon number must correpsond to the sm var being use.
        SWC_L01 corresponds to horizon 0, SWC_L02 corresponds to horizon 1, SWC_L0n corresponds to horizon n-1

    Returns
    -------
    smi_EDO : xarray.DataArray
        The standardised soil moisture index according to the European Drought Observatory
    
    """
    # Select the soil moisture content at L1
    sm_vars = [sm_var for sm_var in mhm_fluxes.variables if 'SWC_L0' in sm_var]
    
    swc = mhm_fluxes[[sm_var]] 
    if sm_var not in sm_vars:
        raise ValueError(f"Variable {sm_var} not found in the mHM dataset. Please select one of the following: {sm_vars}")
    
    # Group the data into weekly averages
    swc_weekly = group_weekly(swc)

    # Select soil field capacity and wilting point at L1
    soil_water_limits = soil_hydr_grid[['L1_soilMoistFC', 'L1_wiltingPoint']].isel(
        L1_LandCoverPeriods=0,  # Remains fixed since there is only one land cover period
        L1_SoilHorizons= nhorizon      # Can be changed to the number of horizons in the mHM model
    ).assign_coords(
        lat=(("lon", "lat"), soil_hydr_grid['L1_domain_lat'].values),
        lon=(("lon", "lat"), soil_hydr_grid['L1_domain_lon'].values)
    )
    
    #check if nhorizon is 1 less than the last digit of the var
    if nhorizon != int(sm_var[-1]) - 1:
        raise ValueError(f"Horizon {nhorizon} does not correspond to variable {sm_var}. Please select the correct horizon.")

    # Rename nrows1 and ncols1 to lat and lon
    soil_water_limits = soil_water_limits.rename({'ncols1': 'lon', 'nrows1': 'lat'})

    #Extract Field capacity and wilting point as arrays and assign the dims of swc
    fc_array = soil_water_limits['L1_soilMoistFC'].values
    wp_array = soil_water_limits['L1_wiltingPoint'].values

    # Create DataArray for field capacity
    fc_da = xr.DataArray(fc_array, dims=('lat', 'lon'), coords={'lat': swc['lat'], 'lon': swc['lon']})

    # Create DataArray for wilting point
    wp_da = xr.DataArray(wp_array, dims=('lat', 'lon'), coords={'lat': swc['lat'], 'lon': swc['lon']})

    # Calculate mean of FC and wilting point
    theta_50 = (fc_da + wp_da) / 2

    # Calculate plant available water (paw)
    #paw = soil_water_limits['L1_soilMoistFC'] - soil_water_limits['L1_wiltingPoint']

    #Calculate Weekly plant available water (paw) scaled to 0-1
    paw_scaled = (swc_weekly - wp_da) / (fc_da - wp_da)

     # Calculate the weekly EDO SMI (According to European Drought Observatory)
    smi_EDO = 1 - 1/(1 +(swc_weekly/theta_50)**6)

    # Return both smi_EDO and paw_scaled in a dictionary
    return {
        'smi_EDO': smi_EDO,
        'paw_scaled': paw_scaled
    }

    #Example usage:
    # to access each of the returned values, use the following syntax:
    #result = calculate_paw_smi(mhm_fluxes, soil_hydr_grid, 'SWC_L01', 0)
    #smi_EDO = result['smi_EDO']
    #paw_scaled = result['paw_scaled']


#==>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def calculate_sma(smi_EDO, reference_period, target_date):
    """
    Calculate the Soil Moisture Anomaly (SMA) for a particular (week, year) pair
    based on the corresponding week in the reference period of choice.

    Parameters
    ----------
    smi_EDO : xarray.DataArray
        The standardised soil moisture index according to the European Drought Observatory
    reference_period : tuple
        A tuple of two datetime objects representing the start and end of the reference period
        e.g. ('1996-01-01', '2016-12-31'). Must include the brackets.
    target_date : str
        A string representing the target date in the format 'mm/dd/yyyy' e.g. '01/15/1997'

    Returns
    -------
    sma : xarray.DataArray
        The soil moisture anomaly for the target date based on the reference period

    """
    #Calculate the weekly Soil Moisture Anomaly (SMA)
    #Reference period 1996-2016
    ref_data = smi_EDO.sel(time=slice(reference_period))

    #Calculate the mean for each week of the reference period
    ref_mean_weekly = ref_data.groupby('week').mean(dim='time')
    #Calculate the standard deviation for each week of the reference period
    ref_std_weekly = ref_data.groupby('week').std(dim='time')
    
    #Calculate the SMA
    #select a particular week and year after the reference period
    
    target_datetime = pd.to_datetime(target_date)
    target_week = (target_datetime.dayofyear - 1) // 7
    target_year = target_datetime.year

    # Filter for year = 1997, dropping everything else
    yeardata = smi_EDO.sel(time=smi_EDO['time'].dt.year == target_year)
    # Now filter for the target week, dropping everything else
    weekdata = yeardata.where(yeardata['week'] == target_week, drop=True)
    #SMA = (SWC - mean) / std
    weekly_sma = (weekdata - ref_mean_weekly.sel(week=target_week)) / ref_std_weekly.sel(week=target_week)

    return weekly_sma

#Example usage:
#result = calculate_paw_smi('SWC_L01', 0)
#smi_EDO = result['smi_EDO']
#paw_scaled = result['paw_scaled']
#calculate_sma(smi_EDO, ('1996-01-01', '2016-12-31'), '01/15/1997')


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def extract_seasonal_totals(ds):
    """
    Compute the seasonal totals of a given DataArray.
    The seasonal totals are computed for each meteorological season (DJF, MAM, JJA, SON).

    Parameters
    ----------
    ds : DataArray
        The input data array.

    Returns
    -------
    seasonal_totals : dict
        A dictionary containing the seasonal totals for each meteorological season, averaged over all years.
    """

    seasonal_totals = {}
    DJF_totals, MAM_totals, JJA_totals, SON_totals = [], [], [], []

    # Group by meteorological seasons (DJF, MAM, JJA, SON)
    for year in np.unique(ds.time.dt.year):
        # DJF: December, January, February (December of the current year, January and February of the next year)
        if (year % 4 == 0):  # Handle leap years for February
            DJF = ds.sel(time=slice(f"{year-1}-12-01", f"{year}-02-29"))
        else:
            DJF = ds.sel(time=slice(f"{year}-12-01", f"{year+1}-02-28"))

        # MAM: March, April, May
        MAM = ds.sel(time=slice(f"{year}-03-01", f"{year}-05-31"))

        # JJA: June, July, August
        JJA = ds.sel(time=slice(f"{year}-06-01", f"{year}-08-31"))

        # SON: September, October, November
        SON = ds.sel(time=slice(f"{year}-09-01", f"{year}-11-30"))

        # Calculate the sum of recharge for each season
        DJF_sum = DJF.sum(dim='time')
        MAM_sum = MAM.sum(dim='time')
        JJA_sum = JJA.sum(dim='time')
        SON_sum = SON.sum(dim='time')

        # Append the seasonal sums to the dictionary

        DJF_totals.append(DJF_sum)
        MAM_totals.append(MAM_sum)
        JJA_totals.append(JJA_sum)
        SON_totals.append(SON_sum)

        #calculate the mean of each season
        DJF_mean = sum(DJF_totals)/len(DJF_totals)
        MAM_mean = sum(MAM_totals)/len(MAM_totals)
        JJA_mean = sum(JJA_totals)/len(JJA_totals)
        SON_mean = sum(SON_totals)/len(SON_totals)

        #seasons
        seasonal_totals['DJF'] = DJF_mean
        seasonal_totals['MAM'] = MAM_mean
        seasonal_totals['JJA'] = JJA_mean
        seasonal_totals['SON'] = SON_mean

    return seasonal_totals