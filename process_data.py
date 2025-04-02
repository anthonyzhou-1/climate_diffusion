import h5py 
import pickle 
from tqdm import tqdm 
import os 

# Custom imports
from common.utils import  get_yaml, dict2namespace
from dataset.datamodule import ClimateDataModule

config_path = "configs/base.yaml"
config = get_yaml(config_path)
config = dict2namespace(config)

config.training.batch_size_per_device = 1

split = "valid"

datamodule = ClimateDataModule(config=config)

if split == "train":
    loader = datamodule.train_dataloader()
    num_samples = 146096
elif split == 'valid':
    loader = datamodule.val_dataloader()
    num_samples = 1460

dset = loader.dataset
dat = dset.dat

nlat = 64
nlon = 128 
nsurface_channels = 8
nlevels = 10
nmulti_channels = 5

path = f"/data/PLASIM/PLASIM_{split}_{num_samples}.h5"
pickle_path = f"/data/PLASIM/PLASIM_{split}_{num_samples}_times.pkl"

with open(pickle_path, 'wb') as handle:
    pickle.dump(dat.time.load(), handle)

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

for i in tqdm(range(num_samples)):
    batch = dset.__getitem__(i)
    surface_feat, multi_level_feat, _, _, day_of_year, hour_of_day = batch

    h5f_u[i] = surface_feat.squeeze()
    h5f_um[i] = multi_level_feat.squeeze()
    hourofdaycoord[i] = hour_of_day.squeeze()
    dayofyearcoord[i] = day_of_year.squeeze()

h5f.close()