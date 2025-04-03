import glob
import xarray as xr
import re
# 1. List all NetCDF files matching the pattern.
all_files = glob.glob('/pscratch/sd/a/ayz2/PLASIM/data_*.nc')

# 2. Sort files numerically by extracting the year from the filename.
def extract_year(filename):
    match = re.search(r'data_(\d+)_sigma\.nc', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Year not found in filename: {filename}")

sorted_files = sorted(all_files, key=extract_year)

# 3. Select only the first 40 years (assuming each file is one year).
# years go from 7 to 111
# train on 12-111, valid on 11
train_files = sorted_files[5:] # 12-111
valid_files = sorted_files[4:5] # 11

# Optionally, print the selected files to verify.
print("Selected files for training set:")
for f in train_files:
    print(f)

print("Selected files for validation set:")
for f in valid_files:
    print(f)

# 4. Open the selected files and combine them along the time dimension.

ds = xr.open_mfdataset(train_files, combine='by_coords')

ds = ds.chunk({'time': 1460, 'lat': 64, 'lon': 128})

# 6. Save the combined dataset to a Zarr store.
ds.to_zarr('/pscratch/sd/a/ayz2/PLASIM/processed_new/PLASIM_train_12-111.zarr', mode='w')