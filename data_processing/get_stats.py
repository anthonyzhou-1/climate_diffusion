import xarray as xr
import os
import numpy as np
import shutil
from matplotlib import pyplot as plt

dat = xr.open_zarr('/pscratch/sd/a/ayz2/PLASIM/processed_new/PLASIM_train_12-111.zarr')
# defined on the surface of earth 
SURFACE_FEATURES = [
    'evap', # lwe_of_water_evaporation
    "mrro", # surface_runoff
    "mrso", # lwe_of_soil_moisture_content
    "pl", # log_surface_pressure
    "pr_12h", # 12-hour accumulated precipitation
    "pr_6h", # 6-hour accumulated precipitation
    "tas", # air_temperature_2m
    "ts", # surface_temperature
]

# defined on many levels of the atmosphere
MULTI_LEVEL_FEATURES = [
    'hus', # specific_humidity
    "ta", # air_temperature
    "ua", # eastward_wind
    "va", # northward_wind
    "zg", # geopotential
]

variables = SURFACE_FEATURES + MULTI_LEVEL_FEATURES

def get_mean_and_std(data):
    print(data)
    # data in shape: [frames, height, width] or [frames, levels, height, width]
    return data.mean(('time', 'lat', 'lon')).to_numpy(), data.std(('time', 'lat', 'lon')).to_numpy()   # level-wise

base_path = "/pscratch/sd/a/ayz2/PLASIM/processed_new"

normalize_mean = {}
normalize_std = {}
for variable in variables:
    print(f'Computing statistics for {variable}')
    mean, std = get_mean_and_std(dat[variable].sel(time=slice('0012', '0111')))
    normalize_mean[variable] = mean
    normalize_std[variable] = std

path_to_dump = os.path.join(base_path, 'norm_stats.npz')
np.savez(path_to_dump,
         normalize_mean=normalize_mean,
         normalize_std=normalize_std)