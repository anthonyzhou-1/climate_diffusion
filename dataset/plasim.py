import torch
import numpy as np
from einops import rearrange
from torch.utils.data import Dataset
import xarray as xr
import pandas as pd
import h5py as h5f
import cftime

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

# constant for all time
CONSTANTS_FEATURES = [
    'lsm', # land_binary_mask
    'sg', # surface_geopotential
    'z0', # surface_roughness_length
]

# constant for each day, but repeat for each year
# Also defined for leap years. Has an extra day, which is 4 more intervals
YEARLY_FEATURES = [
    'rsdt', # TOA (Top of Atmosphere) Incident Shortwave Radiation 
    'sic', # sea_ice_cover
    'sst', # surface_temperature
]

class Normalizer:
    def __init__(self, stat_dict):
        # stat_dict: {feature_name: (mean, std)}
        self.stat_dict = {k: (torch.from_numpy(v[0]).float(), torch.from_numpy(v[1]).float()) if isinstance(v[0], np.ndarray)
        else (torch.tensor(v[0]).float(), torch.tensor(v[1]).float())
                          for k, v in stat_dict.items()}
        # take only first 10 levels of geopotential
        self.stat_dict['zg'] = (self.stat_dict['zg'][0][..., :10], self.stat_dict['zg'][1][..., :10])

    @torch.no_grad()
    def normalize(self, out_dict):
        for k, v in out_dict.items():
            mean, std = self.stat_dict[k]
            mean, std = mean.to(v.device), std.to(v.device)
            if len(v.shape) == 3:
                out_dict[k] = (v - mean.view(1, 1, 1)) / std.view(1, 1, 1)
            elif len(v.shape) == 4:
                out_dict[k] = (v - mean.view(1, 1, 1, -1)) / std.view(1, 1, 1, -1)
            else:
                raise ValueError(f'Invalid shape {v.shape}')

    def denormalize(self, out_dict):
        for k, v in out_dict.items():
            mean, std = self.stat_dict[k]
            mean, std = mean.to(v.device), std.to(v.device)
            if len(v.shape) == 3:   # t nlat nlon
                out_dict[k] = v * std.view(1, 1, 1) + mean.view(1, 1, 1)
            elif len(v.shape) == 4:         # t nlat nlon nlevels
                out_dict[k] = v * std.view(1, 1, 1, -1) + mean.view(1, 1, 1, -1)
            else:
                raise ValueError(f'Invalid shape {v.shape}')

    def batch_denormalize(self, out_dict):
        for k, v in out_dict.items():
            mean, std = self.stat_dict[k]
            mean, std = mean.to(v.device), std.to(v.device)
            if len(v.shape) == 4:    # b t nlat nlon
                out_dict[k] = v * std.view(1, 1, 1, 1) + mean.view(1, 1, 1, 1)
            elif len(v.shape) == 5:     # b t nlat nlon nlevels
                out_dict[k] = v * std.view(1, 1, 1, 1, -1) + mean.view(1, 1, 1, 1, -1)
            else:
                raise ValueError(f'Invalid shape {v.shape}')

    def batch_normalize(self, out_dict):
        for k, v in out_dict.items():
            mean, std = self.stat_dict[k]
            mean, std = mean.to(v.device), std.to(v.device)
            if len(v.shape) == 4:    # b t nlat nlon
                out_dict[k] = (v - mean.view(1, 1, 1, 1)) / std.view(1, 1, 1, 1)
            elif len(v.shape) == 5:     # b t nlat nlon nlevels
                out_dict[k] = (v - mean.view(1, 1, 1, 1, -1)) / std.view(1, 1, 1, 1, -1)
            else:
                raise ValueError(f'Invalid shape {v.shape}')

class PLASIMData(Dataset):
    def __init__(self,
                 data_path,
                 norm_stats_path,
                 boundary_path,
                 surface_vars,
                 multi_level_vars,
                 constant_names,
                 yearly_names,
                 normalize_feature=True,
                 interval=1,  # interval=1 is equal to 6 hours
                 nsteps=2,   # spit out how many consecutive future sequences
                 load_into_memory=False,
                 output_timecoords=False,
                 ):

        self.data_path = data_path  # a zarr file
        # open the data
        dat = xr.open_dataset(self.data_path, engine='zarr')

        self.features_names = surface_vars + multi_level_vars
        self.constant_names = constant_names
        self.yearly_names = yearly_names
        self.normalize_feature = normalize_feature
        self.output_timecoords = output_timecoords

        # this assumes that the normalization statistics are stored in the same directory as the data
        self.normalizer = Normalizer(self.load_norm_stats(norm_stats_path))

        self.interval = interval
        self.nsteps = nsteps

        # load the constants, in shape (nlat, nlon, nconstants) or (ntime, nlat, nlon, nyearly)
        self.constants, self.yearly_constants, self.leap_yearly_constants = self.load_constants(boundary_path)

        print('Surface variables:', surface_vars)
        print('Multi-level variables:', multi_level_vars)
        print('Constant variables:', constant_names)
        print('Yearly variables:', yearly_names)
        self.surface_vars = surface_vars
        self.multi_level_vars = multi_level_vars

        # get the time stamps
        time_coords = dat.time.values # array of cftime objects
        start_time_coords = time_coords
        # filter out those will be out of bound
        if nsteps > 0:
            start_time_coords = start_time_coords[:-(interval * nsteps)]
        else:
            start_time_coords = start_time_coords[:]

        # keep in range data
        self.dat = dat.sel(time=time_coords)
        if load_into_memory:
            self.dat.load()
        self.time_coords = time_coords
        self.start_time_coords = start_time_coords

        # if we treat this as initial value problem, how many initial values are there?
        self.nstamps = len(start_time_coords)

    def __len__(self):
        return self.nstamps

    def get_var_names(self):
        return self.surface_vars, self.multi_level_vars, self.constant_names, self.yearly_names

    def load_norm_stats(self, norm_stat_path):
        stat_dict = {}
        with np.load(norm_stat_path, allow_pickle=True) as f:
            normalize_mean, normalize_std = f['normalize_mean'].item(), f['normalize_std'].item()
            for feature_name in self.features_names:
                assert feature_name in normalize_mean.keys(), f'{feature_name} not in {norm_stat_path}'
                stat_dict[feature_name] = (normalize_mean[feature_name], normalize_std[feature_name])
        return stat_dict
    
    def load_constants(self, boundary_path):
        # load constants into local memory. About 300 Mb
        boundary_file = h5f.File(boundary_path, 'r')
        constants_dict = {}
        for constant in self.constant_names:
            assert constant in boundary_file.keys(), f'{constant} not in the boundary file'
            constant_values = boundary_file[constant][:] # (nlat, nlon) for constants, (time, nlat, lon) for yearly
            # scale to [-1, 1]
            scaled_constant = 2 * (constant_values - constant_values.min()) / (constant_values.max() - constant_values.min()) - 1
            constants_dict[constant] = torch.from_numpy(scaled_constant).float()

        yearly_constants_dict = {}
        for constant in self.yearly_names:
            assert constant in boundary_file.keys(), f'{constant} not in the boundary file'
            constant_values = boundary_file[constant][:] # (nlat, nlon) for constants, (time, nlat, lon) for yearly
            # scale to [-1, 1]
            scaled_constant = 2 * (constant_values - constant_values.min()) / (constant_values.max() - constant_values.min()) - 1
            yearly_constants_dict[constant] = torch.from_numpy(scaled_constant).float()

        leap_yearly_names = [f'{constant}_leap' for constant in self.yearly_names]
        leap_yearly_constants_dict = {}
        for constant in leap_yearly_names:
            assert constant in boundary_file.keys(), f'{constant} not in the boundary file'
            constant_values = boundary_file[constant][:] # (nlat, nlon) for constants, (time, nlat, lon) for yearly
            # scale to [-1, 1]
            scaled_constant = 2 * (constant_values - constant_values.min()) / (constant_values.max() - constant_values.min()) - 1
            leap_yearly_constants_dict[constant] = torch.from_numpy(scaled_constant).float()
        boundary_file.close()

        constants = torch.stack([constants_dict[k] for k in self.constant_names], dim=-1) # (nlat, nlon, nconstants)
        yearly_constants = torch.stack([yearly_constants_dict[k] for k in self.yearly_names], dim=-1) # (ntime, nlat, nlon, nyearly)
        leap_yearly_constants = torch.stack([leap_yearly_constants_dict[k] for k in leap_yearly_names], dim=-1) # (ntime_leap, nlat, nlon, nyearly)
        return constants, yearly_constants, leap_yearly_constants
    
    def __getitem__(self, idx):
        # fetch time coord first
        start_time_idx = idx
        start_time = self.start_time_coords[start_time_idx]
        # find start time coord in time coords
        time_coord = self.time_coords[start_time_idx: start_time_idx + self.interval * (self.nsteps + 1): self.interval]
        assert time_coord[0] == start_time, f'{time_coord[0]} != {start_time}'

        dat_slice = self.dat.sel(time=time_coord) # time, nlat, nlon, etc. 
        # get the feature data
        features = dat_slice[self.features_names] # time, nlat, nlon, etc.

        # prepare the feature dict
        feature_dict = {k: rearrange(torch.from_numpy(v.values.T), 'nlon nlat nlevel nt -> nt nlat nlon nlevel')
                        if k in self.multi_level_vars else
                            rearrange(torch.from_numpy(v.values.T), 'nlon nlat nt -> nt nlat nlon')
                        for k, v in features.items()}
        
        # take 1st 10 levels of geopotential
        feature_dict['zg'] = feature_dict['zg'][..., :10]

        # normalize the feature
        if self.normalize_feature:
            self.normalizer.normalize(feature_dict)

        # pack the feature into surface and upper air
        surface_feat = []
        multi_level_feat = []

        for k, v in feature_dict.items():
            if k in self.surface_vars:
                surface_feat.append(v)
            elif k in self.multi_level_vars:
                multi_level_feat.append(v)
            else:
                raise ValueError(f'Unknown feature {k}')

        # stack the surface features
        surface_feat = torch.stack(surface_feat, dim=-1) # shape (nt, nlat, nlon, surface_channels)
        multi_level_feat = torch.stack(multi_level_feat, dim=-1) # shape (nt, nlat, nlon, nlevel, multi_level_channels)

        # get temporal coords
        timestamp = [pd.Timestamp(t.strftime()) for t in time_coord]

        # check if all time coords are during a leap year
        leap_years = [cftime.is_leap_year(time_coord_i.year, 'proleptic_gregorian') for time_coord_i in time_coord]

        # Extract day of the year and hour of the day
        day_of_year = []
        for i in range(len(time_coord)):
            if leap_years[i]:
                num_days = 366
            else:
                num_days = 365
            day_of_year.append(min(timestamp[i].day_of_year / num_days, 1))

        hour_of_day = [t.hour / 24 for t in timestamp]
        
        yearly_constants = []
        for i in range(len(time_coord)):
            year_string = time_coord[i].strftime('%Y')
            num_hours = cftime.date2num(time_coord[i], f'hours since {year_string}-01-01 00:00:00', calendar='proleptic_gregorian')
            yearly_idx = int(num_hours // 6) # 6 hours per interval
            if leap_years[i]:
                yearly_constants_i = self.leap_yearly_constants[yearly_idx] # (nlat, nlon, nyearly)
            else:
                yearly_constants_i = self.yearly_constants[yearly_idx] # (nlat, nlon, nyearly)
            yearly_constants.append(yearly_constants_i)
        
        yearly_constants = torch.stack(yearly_constants, dim=0) # (nt, nlat, nlon, nyearly)

        if not self.output_timecoords:
            return surface_feat, multi_level_feat,\
                   self.constants.clone(), yearly_constants, torch.Tensor(day_of_year), torch.Tensor(hour_of_day)
        else:
            return surface_feat, multi_level_feat,\
                   self.constants.clone(), yearly_constants, torch.Tensor(day_of_year), torch.Tensor(hour_of_day), timestamp











