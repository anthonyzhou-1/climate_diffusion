# Climate Diffusion

## Requirements
To install dependencies:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install lightning -c conda-forge
pip install wandb h5py einops matplotlib torch-harmonics 
pip install "xarray[complete]" 
```

The PDE-Refiner baseline has extra dependencies:
```
conda install -c conda-forge diffusers
```

## Datasets

Data is organized in a zarr format. 

## Training
Train script will expect a logging directory, paths to the dataset, and a WandB instance.
```
python train.py --config=configs/base.yaml
```