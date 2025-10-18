#%%
import pandas as pd
import numpy as np
import xarray as xr
import os

# %% Read CSV file
src_path = r"W:\VUB\_main_research\data\RMI"
os.chdir(src_path)

names = ['DATE', 'PIXEL_ID', 'PIXEL_LON_CENTER', 'PIXEL_LAT_CENTER', 
         'TEMPERATURE MAX', 'TEMPERATURE MIN', 'PRECIPITATION (mm)', 'ET0 (mm)', 'global solar radiation (W/m2)', 'wind speed (m/s)']

# Read the data with appropriate encoding
clim_data = pd.read_csv("pdg1487.csv", sep=";", encoding="ISO-8859-1", skiprows=1, names=names)

# Prepare data
clim_data.set_index(['DATE'], inplace=True)
clim_data.sort_index(inplace=True)

# %% Prepare data for NetCDF writing

variables = ['PIXEL_LON_CENTER', 'PIXEL_LAT_CENTER', "TEMPERATURE MAX", "TEMPERATURE MIN", "PRECIPITATION (mm)", "ET0 (mm)"]
climate_variables = ['PIXEL_LON_CENTER', 'PIXEL_LAT_CENTER', 'tmax', 'tmin', 'pr', 'ETo']

clim_data_subset = clim_data[variables]
clim_data_subset.columns = climate_variables

# Unique sorted coordinates
lon = np.sort(clim_data_subset['PIXEL_LON_CENTER'].unique())
lat = np.sort(clim_data_subset['PIXEL_LAT_CENTER'].unique())[::-1]
time = pd.to_datetime(clim_data_subset.index.unique(), format='%Y-%m-%d')

# Define NetCDF file to write to
output_file = "rmi_climate_data.nc"

# Create an empty dataset template for writing
ds_template = xr.Dataset(
    {
        "tmax": (["lat", "lon"], np.full((len(lat), len(lon)), np.nan, dtype=np.float32)),
        "tmin": (["lat", "lon"], np.full((len(lat), len(lon)), np.nan, dtype=np.float32)),
        "pr":   (["lat", "lon"], np.full((len(lat), len(lon)), np.nan, dtype=np.float32)),
        "ET0":  (["lat", "lon"], np.full((len(lat), len(lon)), np.nan, dtype=np.float32)),
    },
    coords={
        "lat": lat,
        "lon": lon,
    }
)

# Initialize a NetCDF file
ds_template.to_netcdf(output_file, mode="w")

# %% Process data in chunks of 100 time steps
chunk_size = 100  # Number of time steps per chunk

for i in range(0, len(time), chunk_size):
    # Determine the time range for this chunk
    time_chunk = time[i:i + chunk_size]
    print(f"Processing time steps {i + 1} to {min(i + chunk_size, len(time))} ({len(time_chunk)} steps)")

    # Initialize grids for this chunk
    tmax_grid = np.full((len(time_chunk), len(lat), len(lon)), np.nan, dtype=np.float32)
    tmin_grid = np.full((len(time_chunk), len(lat), len(lon)), np.nan, dtype=np.float32)
    pr_grid = np.full((len(time_chunk), len(lat), len(lon)), np.nan, dtype=np.float32)
    ET0_grid = np.full((len(time_chunk), len(lat), len(lon)), np.nan, dtype=np.float32)
    
    # Process each time step within the chunk
    for t_idx, t in enumerate(time_chunk):
        df_time = clim_data_subset.loc[clim_data_subset.index == t.strftime('%Y-%m-%d')]
        
        for _, row in df_time.iterrows():
            lon_idx = np.where(lon == row['PIXEL_LON_CENTER'])[0][0]
            lat_idx = np.where(lat == row['PIXEL_LAT_CENTER'])[0][0]
            tmax_grid[t_idx, lat_idx, lon_idx] = row['tmax']
            tmin_grid[t_idx, lat_idx, lon_idx] = row['tmin']
            pr_grid[t_idx, lat_idx, lon_idx] = row['pr']
            ET0_grid[t_idx, lat_idx, lon_idx] = row['ETo']
    
    ds_chunk = xr.Dataset(
    {
        "tmax": (["time", "lat", "lon"], tmax_grid),
        "tmin": (["time", "lat", "lon"], tmin_grid),
        "pr":   (["time", "lat", "lon"], pr_grid),
        "ET0":  (["time", "lat", "lon"], ET0_grid),
    },
    coords={
        "time": np.array(time_chunk),  # Ensure time_chunk is a 1D array
        "lat": lat,
        "lon": lon,
    }
)

    
    # Append chunk data to the NetCDF file
    ds_chunk.to_netcdf(output_file, mode="a", unlimited_dims=["time"])

print("Processing complete. NetCDF file saved as:", output_file)

# %%
