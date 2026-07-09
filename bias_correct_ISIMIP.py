#py_climate env
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import process_ISIMIP as pISIMIP

"""
-----------------------------
1. READ and Group ISIMIP data
-----------------------------
"""

#directories
src = "/vscmnt/brussel_pixiu_data/_data_brussel/vo/000/bvo00033/vsc10530/mHM_model/mhm/prepare_ISIMIP_to_mHM_metdata/"

src_isimip = f"{src}/ISIMIP_forcing_data/"

#source for EOBS data (observed)
src_obs = f"{src}/e_OBS_data/"

#destination for bias-corrected ISIMIP data
dest_isimip = f"{src}/bias_corrected_ISIMIP_data/"

#Data already clipped to domain boundaries
tmax_EOBS = xr.open_dataset(f'{src_obs}/tmax.nc')
tmin_EOBS = xr.open_dataset(f'{src_obs}/tmin.nc')
pr_EOBS = xr.open_dataset(f'{src_obs}/pre.nc')


models=(
    "UKESM1-0-LL",
    "MRI-ESM2-0",
    "MPI-ESM1-2-HR",
    "GFDL-ESM4",
    "IPSL-CM6A-LR",
    "CNRM-CM6-1",
    "CanESM5",
    "MIROC6",
    "TaiESM1",
    "EC-Earth3"
)

#select only scenarios to model
scenarios = ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
variables = ['pr', 'tasmin', 'tasmax']

# Process Merged ISIMIP scenario files


#slice future periods by time window of interest
"""Customize as necessary"""
historical_periods = slice("1971-01-01", "2010-12-31")
future_periods = slice("2061-01-01", "2100-12-31")

#store ISIMIP data in a dictionary
isimip_data = {}

#Open the datasets for each model and scenario and slice accordingly
for model in models:
    for scenario in scenarios:
        for variable in variables:
            file_path = f"{src_isimip}/{model}/{scenario}/{model}_{scenario}_{variable}_merged.nc"
            if glob.glob(file_path):
                data = xr.open_dataset(file_path)
                if scenario == 'historical':
                    isimip_data[f'{model}_{scenario}_{variable}'] = data.sel(time=historical_periods)
                else:
                    isimip_data[f'{model}_{scenario}_{variable}'] = data.sel(time=future_periods)


"""
-----------------------------
2.  REGRID ISIMIP Data to match resolution of Observed E-OBS data
-----------------------------
"""

"""RAINFALL"""
#regrid climate data to EOBS grid
# ----------------------------
# Settings you can tweak
# ----------------------------
REGRID_METHOD = "bilinear"          # "bilinear" for rates; use "conservative_normed" for totals
VAR_NAME = "pr"              # precipitation variable key in your isimip_data
VAR_UNITS = "mm/day"         # CF-ish units string
VAR_STDNAME = "precipitation_flux"
long_name = "Regridded precipitation rate"
# -----------------------------------

obs_pre = pISIMIP.select_ref_hist(pr_EOBS, 'pre', '1971-01-01', '2010-12-31') #Overlap with historical_period for ISIMIP data

isimip_regridded_pr = pISIMIP.regrid_ISIMIP_to_obs(isimip_data, obs_pre, models, scenarios, REGRID_METHOD,
                                             VAR_NAME, VAR_UNITS, VAR_STDNAME, long_name)


#Get rid of empty rows and columns in the regridded data
# -----------------------------------
isimip_hiRES_pr = {}

#mask and drop empty rows/cols
for name, data in isimip_regridded_pr.items():

        #cells that are valid at least once over time
        valid = data.notnull().any("time")

        # mask and drop empty rows/cols
        data_valid = data.where(valid).dropna("lat", how="all").dropna("lon", how="all")     

        isimip_hiRES_pr[name] = data_valid


"""TEMPERATURE"""
#regrid climate data to EOBS grid
# ----------------------------
# Settings you can tweak
# ----------------------------
REGRID_METHOD = "bilinear"          # "bilinear" for rates; use "conservative_normed" for totals
VAR_NAME = "tasmax"              # precipitation variable key in your isimip_data
VAR_UNITS = "degree-Celsius"         # CF-ish units string
VAR_STDNAME = "Maximum daily air temperature"
long_name = "Regridded maximum daily air temperature"
# -----------------------------------

obs_tmax = pISIMIP.select_ref_hist(tmax_EOBS, 'tmax', '1971-01-01', '2010-12-31') #Overlap with historical_period for ISIMIP data

isimip_regridded_tasmax = pISIMIP.regrid_ISIMIP_to_obs(isimip_data, obs_tmax, models, scenarios, REGRID_METHOD,
                                             VAR_NAME, VAR_UNITS, VAR_STDNAME, long_name)


#Get rid of empty rows and columns in the regridded data
# -----------------------------------
isimip_hiRES_tmax= {}

#mask and drop empty rows/cols
for name, data in isimip_regridded_tasmax.items():

        #cells that are valid at least once over time
        valid = data.notnull().any("time")

        # mask and drop empty rows/cols
        data_valid = data.where(valid).dropna("lat", how="all").dropna("lon", how="all")

        #rename tasmax, tasmin variable to tmax, tmin
        #data_valid = data_valid.rename({'tasmax': 'tmax'})


        isimip_hiRES_tmax[name] = data_valid


#regrid climate data to EOBS grid
# ----------------------------
# Settings you can tweak
# ----------------------------
REGRID_METHOD = "bilinear"          # "bilinear" for rates; use "conservative_normed" for totals
VAR_NAME = "tasmin"              # variable key in isimip_data
VAR_UNITS = "degree-Celsius"         # CF-ish units string
VAR_STDNAME = "Minimum daily air temperature"
long_name = "Regridded minimum daily air temperature"
# -----------------------------------

obs_tmin = pISIMIP.select_ref_hist(tmin_EOBS, 'tmin', '1971-01-01', '2010-12-31')

isimip_regridded_tasmin = pISIMIP.regrid_ISIMIP_to_obs(isimip_data, obs_tmin, models, scenarios, REGRID_METHOD,
                                             VAR_NAME, VAR_UNITS, VAR_STDNAME, long_name)


#Get rid of empty rows and columns in the regridded data
# -----------------------------------
isimip_hiRES_tmin= {}

#mask and drop empty rows/cols
for name, data in isimip_regridded_tasmin.items():

        #cells that are valid at least once over time
        valid = data.notnull().any("time")

        # mask and drop empty rows/cols
        data_valid = data.where(valid).dropna("lat", how="all").dropna("lon", how="all")

        #rename tasmax, tasmin variable to tmax, tmin
        #data_valid = data_valid.rename({'tasmin': 'tmin'})

        isimip_hiRES_tmin[name] = data_valid



"""
#-------------------------------------------------------------------------------------------------------------
#4. BIAS CORRECT ISIMIP Data to match Observed E-OBS data
#--------------------------------------------------------------------------------------------------------------
"""
#### Precipitation
future_scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
method= 'qdm'  # Choose 'qdm' for Quantile Delta Mapping or 'eqm' for Empirical Quantile Mapping
kind = "*"  # Use "+" for temperature or "*" for precipitation
VAR_NAME, VAR_UNITS, VAR_STDNAME= 'pr', 'mm/day', 'precipitation_flux'

# Bias correct the regridded ISIMIP data
"""QDM"""
isimip_bc_precip = pISIMIP.bias_correct_ISIMIP(isimip_hiRES_pr, obs_pre, models,
                                     future_scenarios, method=method, VAR_NAME=VAR_NAME, VAR_UNITS=VAR_UNITS, VAR_STDNAME=VAR_STDNAME, kind=kind)

"""EDM"""
# isimip_bc_EQM = pISIMIP.bias_correct_ISIMIP(isimip_regridded_pr, obs_precip, models,
#                                      future_scenarios, method=method)

##--------------------------------------------------------------------------------------------------------------------------------------------------------

#####  Tmax
method= 'qdm'  # Choose 'qdm' for Quantile Delta Mapping or 'eqm' for Empirical Quantile Mapping
kind = "+"  # Use "+" for temperature or "*" for precipitation
VAR_NAME, VAR_UNITS, VAR_STDNAME= 'tasmax', 'degree_Celsius', 'max. daily temperature'

# Bias correct the regridded ISIMIP data
"""QDM"""

isimip_bc_tmax = pISIMIP.bias_correct_ISIMIP(isimip_hiRES_tmax, obs_tmax, models,
                                     future_scenarios, method=method, VAR_NAME=VAR_NAME, VAR_UNITS=VAR_UNITS, VAR_STDNAME=VAR_STDNAME, kind=kind)

"""EDM"""
# isimip_bc_EQM = pISIMIP.bias_correct_ISIMIP(isimip_regridded_pr, obs, models,
#                                      future_scenarios, method=method)
##--------------------------------------------------------------------------------------------------------------------------------------------------------
#####  Tmin
method= 'qdm'  # Choose 'qdm' for Quantile Delta Mapping or 'eqm' for Empirical Quantile Mapping
kind = "+"  # Use "+" for temperature or "*" for precipitation
VAR_NAME, VAR_UNITS, VAR_STDNAME= 'tasmin', 'degree_Celsius', 'min. daily temperature'

# Bias correct the regridded ISIMIP data
"""QDM"""

isimip_bc_tmin = pISIMIP.bias_correct_ISIMIP(isimip_hiRES_tmin, obs_tmin, models,
                                     future_scenarios, method=method, VAR_NAME=VAR_NAME, VAR_UNITS=VAR_UNITS, VAR_STDNAME=VAR_STDNAME, kind=kind)

"""EDM"""
# isimip_bc_EQM = pISIMIP.bias_correct_ISIMIP(isimip_regridded_pr, obs, models,
#                                      future_scenarios, method=method)
##--------------------------------------------------------------------------------------------------------------------------------------------------------

#calculate tasavg
# -----------------------------------

isimip_bc_tavg = {}
# Loop through each model and scenario to calculate tasavg
scenarios = ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585']

for model in models:
    for scenario in scenarios:
        tasmin = isimip_bc_tmin[f'{model}_{scenario}_tasmin']['tasmin']
        tasmax = isimip_bc_tmax[f'{model}_{scenario}_tasmax']['tasmax']
        tasavg = (tasmin + tasmax) / 2

        #change to xrDataset
        tasavg = tasavg.to_dataset(name='tavg')

        attributes = {
            'units': 'degree_Celsius',
            'standard_name': 'Average daily air temperature',
            'long_name': f'Average daily air temperature for {model} {scenario}'
             
        }

        tasavg.attrs.update(attributes)

        isimip_bc_tavg[f'{model}_{scenario}_tavg'] = tasavg


# RENAME VARIABLES TO mHM format
# precip → pre
isimip_bc_precip = {
    key.replace("pr", "pre"): data.rename({"pr": "pre"})
    for key, data in isimip_bc_precip.items()
}

# tmax → tmax
isimip_bc_tmax = {
    key.replace("tasmax", "tmax"): data.rename({"tasmax": "tmax"})
    for key, data in isimip_bc_tmax.items()
}

# tmin → tmin
isimip_bc_tmin = {
    key.replace("tasmin", "tmin"): data.rename({"tasmin": "tmin"})
    for key, data in isimip_bc_tmin.items()
}


#Export the bias-corrected ISIMIP data to NetCDF files
###     Export the bias-corrected data to NetCDF files
# -----------------------------------
export_scenarios = ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
variable = 'tavg'

#variable_dict combinations

variable_dict = {
    'pre': isimip_bc_precip,
    'tmax': isimip_bc_tmax,
    'tmin': isimip_bc_tmin,
    'tavg': isimip_bc_tavg
}

#create if not exists
if not os.path.exists(dest_isimip):
    os.makedirs(dest_isimip)

for variable, dataset_dict in variable_dict.items():
     #export the regridded tasmin data
     pISIMIP.export_scenarios_to_netcdf(dataset_dict, dest_isimip, export_scenarios, models, variable)