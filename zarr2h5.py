import h5py 
import pickle 
from tqdm import tqdm 
import os 

# Custom imports
from dataset.plasim import PLASIMData, SURFACE_FEATURES, MULTI_LEVEL_FEATURES, CONSTANTS_FEATURES, YEARLY_FEATURES

split = "train"
#train_path = "/data/PLASIM/PLASIM/PLASIM_train_12-111.zarr"
#valid_path = "/data/PLASIM/PLASIM/PLASIM_valid_11.zarr"
#norm_stats_path = "/data/PLASIM/PLASIM/norm_stats.npz"
#boundary_path = "/data/PLASIM/PLASIM/boundary_vars.h5"

train_path = "/pscratch/sd/a/ayz2/PLASIM/processed/PLASIM_train_12-111.zarr"
valid_path = "/pscratch/sd/a/ayz2/PLASIM/processed/PLASIM_valid_11.zarr"
norm_stats_path = "/pscratch/sd/a/ayz2/PLASIM/processed/norm_stats.npz"
boundary_path = "/pscratch/sd/a/ayz2/PLASIM/processed/boundary_vars.h5"

if split == "train":
    chunk_start = 0
    chunk_size = 397
    chunk_end = chunk_size
    num_samples = 146096
    num_iters = num_samples // chunk_size
    dset = PLASIMData(data_path=train_path,
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
    dset = PLASIMData(data_path=valid_path,
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
nlevels = 10
nmulti_channels = 5

path = f"/pscratch/sd/a/ayz2/PLASIM/processed/PLASIM_{split}_{num_samples}.h5"

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
        dset = PLASIMData(data_path=train_path,
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
if split == "train":
    dset = PLASIMData(data_path=train_path,
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
elif split == 'valid':
    dset = PLASIMData(data_path=valid_path,
                    norm_stats_path=norm_stats_path,
                    surface_vars=SURFACE_FEATURES,
                    boundary_path=boundary_path,
                    multi_level_vars=MULTI_LEVEL_FEATURES,
                    constant_names=CONSTANTS_FEATURES,
                    yearly_names=YEARLY_FEATURES,
                    normalize_feature=False,
                    load_into_memory=False,
                    nsteps=0,
                    chunk_range=[0, num_samples])
    
dat = dset.dat
pickle_path = f"/pscratch/sd/a/ayz2/PLASIM/processed/PLASIM_{split}_{num_samples}_times.pkl"
with open(pickle_path, 'wb') as handle:
    pickle.dump(dat.time.load(), handle)

