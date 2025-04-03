import h5py 
import pickle 
from tqdm import tqdm 
import os 
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
 

                 split='train',
 
                 load_into_memory=False,
 
                 output_timecoords=False,
 

                 chunk_range = [0, -1]
 
                 ):
 

 
        self.data_path = data_path  # a zarr file
 

        # open the data
 

        if split == 'train': # Manually chunk the dataset. Each chunk is around 70 MB
 

            # We know that we will always access all lat/lon and levels at each call, therefore chunk along time dim and combine others
 

            dat = xr.open_dataset(self.data_path, engine='zarr', use_cftime=True) #, chunks={'time': 23, 'plev': 13, 'lev': 10, 'lat': 64, 'lon': 128})
 

        else:
 

            dat = xr.open_dataset(self.data_path, engine='zarr', use_cftime=True) # doesn't matter for val since loading entire array into memory
 

        dat = xr.open_dataset(self.data_path, engine='zarr', use_cftime=True) # doesn't matter for val since loading entire array into memory
 
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
 
        self.surface_vars = surface_vars
 
        self.multi_level_vars = multi_level_vars

        # get the time stamps
 
        time_coords = dat.time.values # array of cftime objects
        print(len(time_coords))
 
        start_time_coords = time_coords
 
        # filter out those will be out of bound
 
        if nsteps > 0:
 
            start_time_coords = start_time_coords[:-(interval * nsteps)]
 
        else:
 

            start_time_coords = start_time_coords[:]
 

            start_time_coords = start_time_coords[chunk_range[0]:chunk_range[-1]]
 

        time_coords = start_time_coords
 
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
            return surface_feat, multi_level_feat, self.constants.clone(), yearly_constants, torch.Tensor(day_of_year), torch.Tensor(hour_of_day)
        else:
            return surface_feat, multi_level_feat, self.constants.clone(), yearly_constants, torch.Tensor(day_of_year), torch.Tensor(hour_of_day), timestamp

split = "train"
#train_path = "/data/PLASIM/PLASIM/PLASIM_train_12-111.zarr"
#valid_path = "/data/PLASIM/PLASIM/PLASIM_valid_11.zarr"
#norm_stats_path = "/data/PLASIM/PLASIM/norm_stats.npz"
#boundary_path = "/data/PLASIM/PLASIM/boundary_vars.h5"

train_path = "/pscratch/sd/a/ayz2/PLASIM/processed_new/PLASIM_train_12-111.zarr"
valid_path = "/pscratch/sd/a/ayz2/PLASIM/processed_new/PLASIM_valid_11.zarr"
norm_stats_path = "/pscratch/sd/a/ayz2/PLASIM/processed/norm_stats.npz"
boundary_path = "/pscratch/sd/a/ayz2/PLASIM/processed/boundary_vars.h5"

if split == "train":
    chunk_start = 0
    chunk_size = 397
    chunk_end = chunk_size
    num_samples = 146096
    num_iters = num_samples // chunk_size
    data_path = train_path
    dset = PLASIMData(data_path=data_path,
                      norm_stats_path=norm_stats_path,
                      surface_vars=SURFACE_FEATURES,
                      boundary_path=boundary_path,
                      multi_level_vars=MULTI_LEVEL_FEATURES,
                      constant_names=CONSTANTS_FEATURES,
                      yearly_names=YEARLY_FEATURES,
                      normalize_feature=False,
                      nsteps=0,
                      load_into_memory=True,
                      chunk_range=[chunk_start, chunk_end])
elif split == 'valid':
    num_samples = 1460
    chunk_size = num_samples
    num_iters = 1
    data_path = valid_path
    dset = PLASIMData(data_path=data_path,
                    norm_stats_path=norm_stats_path,
                    surface_vars=SURFACE_FEATURES,
                    boundary_path=boundary_path,
                    multi_level_vars=MULTI_LEVEL_FEATURES,
                    constant_names=CONSTANTS_FEATURES,
                    yearly_names=YEARLY_FEATURES,
                    normalize_feature=False,
                    load_into_memory=True,
                    nsteps=0,
                    chunk_range=[0, num_samples])

dat = dset.dat

nlat = 64
nlon = 128 
nsurface_channels = 8
nlevels = 13
nmulti_channels = 5

path = f"/pscratch/sd/a/ayz2/PLASIM/processed_new/PLASIM_{split}_{num_samples}.h5"

# check if path exists
if os.path.exists(path):
    os.remove(path)
    print(f'File {path} is deleted.')
else:
    print(f'No file {path} exists yet.')

h5f = h5py.File(path, 'a')

dataset = h5f.create_group(split)

h5f_u = dataset.create_dataset(f'surface', (num_samples, nlat, nlon, nsurface_channels), dtype='f4')
h5f_um = dataset.create_dataset(f'multilevel', (num_samples, nlat, nlon, nlevels, nmulti_channels), dtype='f4')
hourofdaycoord = dataset.create_dataset(f'hour', (num_samples), dtype='f4')
dayofyearcoord = dataset.create_dataset(f'day', (num_samples), dtype='f4')
latcoord = dataset.create_dataset(f'lat', (nlat), dtype='f4')
loncoord = dataset.create_dataset(f'lon', (nlon), dtype='f4')

latcoord[:] = dat.lat.load()
loncoord[:] = dat.lon.load()

for j in tqdm(range(num_iters)):
    for i in range(chunk_size):
        batch = dset.__getitem__(i)
        surface_feat, multi_level_feat, _, _, day_of_year, hour_of_day = batch

        save_idx = j*chunk_size + i
        h5f_u[save_idx] = surface_feat.squeeze()
        h5f_um[save_idx] = multi_level_feat.squeeze()
        hourofdaycoord[save_idx] = hour_of_day.squeeze()
        dayofyearcoord[save_idx] = day_of_year.squeeze()
    
    if num_iters > 1:
        # reload dset
        chunk_start = chunk_start + chunk_size
        chunk_end = chunk_end + chunk_size
        #print(f"Reloading dset with chunk_start={chunk_start} and chunk_end={chunk_end}")
        dset = PLASIMData(data_path=data_path,
                    norm_stats_path=norm_stats_path,
                    surface_vars=SURFACE_FEATURES,
                    boundary_path=boundary_path,
                    multi_level_vars=MULTI_LEVEL_FEATURES,
                    constant_names=CONSTANTS_FEATURES,
                    yearly_names=YEARLY_FEATURES,
                    normalize_feature=False,
                    nsteps=0,
                    load_into_memory=True,
                    chunk_range=[chunk_start, chunk_end])

h5f.close()

# reload entire dset to dump all time coords
dset = PLASIMData(data_path=data_path,
                    norm_stats_path=norm_stats_path,
                    surface_vars=SURFACE_FEATURES,
                    boundary_path=boundary_path,
                    multi_level_vars=MULTI_LEVEL_FEATURES,
                    constant_names=CONSTANTS_FEATURES,
                    yearly_names=YEARLY_FEATURES,
                    normalize_feature=False,
                    nsteps=0,
                    load_into_memory=False,
                    chunk_range=[0, num_samples])

    
dat = dset.dat
pickle_path = f"/pscratch/sd/a/ayz2/PLASIM/processed_new/PLASIM_{split}_{num_samples}_times.pkl"
with open(pickle_path, 'wb') as handle:
    pickle.dump(dat.time.load(), handle)

