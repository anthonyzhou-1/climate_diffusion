import argparse
import torch
import os
import yaml
from einops import rearrange

from matplotlib import pyplot as plt

def plot_result_2d(y_pred, y, filename, num_t=6, cmap='twilight_shifted'):
    # y in shape [t h w], y_pred in shape [t h w]

    t_total, h, w = y_pred.shape

    dt = t_total // num_t
    fig, axs = plt.subplots(2, num_t, figsize=(num_t*6, 6))

    y_pred = y_pred[::dt]
    y = y[::dt]

    vmin = y.min()
    vmax = y.max()

    for i in range(num_t):
        im0 = axs[0][i].imshow(y[i], vmin=vmin, vmax=vmax,cmap=cmap)
        im1 = axs[1][i].imshow(y_pred[i], vmin=vmin, vmax=vmax, cmap=cmap)

        # set the title
        axs[0][i].set_title(f"True t={i*dt}")
        axs[1][i].set_title(f"Pred t={i*dt}")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im0, cax=cbar_ax)
    # save the figure
    plt.savefig(filename, dpi=300)
    plt.close()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_yaml(path):
    with open(path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def save_yaml(config, path):
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def assemble_input(surface_input, multilevel_input):
    multilevel_collapsed = rearrange(multilevel_input, 'b nlat nlon nlevel c -> b nlat nlon (nlevel c)')
    model_input = torch.cat([surface_input, multilevel_collapsed], dim=-1) # b nlat nlon (c + nlevel*c)
    return model_input

def assemble_grid_params(constants, yearly_constants, t):
    # constants in shape b nlat nlon c, yearly_constants in shape b t nlat nlon c
    yearly_constants_t = yearly_constants[:, t] # b nlat nlon c 
    grid_params = torch.cat([constants, yearly_constants_t], dim=-1) # b nlat nlon (c + c)
    return grid_params

def assemble_scalar_params(day_of_year, hour_of_day, t):
    # day of year in shape b t, hour of day in shape b t
    return torch.cat([day_of_year[:, t].unsqueeze(1), hour_of_day[:, t].unsqueeze(1)], dim=1) # b 2

def disassemble_input(assembled_input, num_levels, num_surface_channels):
    surface_input = assembled_input[..., :num_surface_channels]
    multilevel_input = rearrange(assembled_input[..., num_surface_channels:], 'b nlat nlon (nlevel c) -> b nlat nlon nlevel c', nlevel=num_levels)
    return surface_input, multilevel_input

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace