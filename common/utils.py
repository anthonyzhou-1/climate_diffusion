import argparse
import torch
import os
import yaml
from einops import rearrange

from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import ImageGrid

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=42, help='Global seed')
    parser.add_argument('--devices', nargs='+', help='<Required> Set flag', default=[])
    args = parser.parse_args()

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    # copy the config file to the log_dir
    return args, config

def plot_result_2d(y, y_pred, filename,
                   num_vis=3, num_t=6, cmap='twilight'):
    matplotlib.use('Agg')

    # visualize the result in 2D rectangular map
    # y and y_pred should be in shape [Nsample, time, lat, lon]
    # we randomly pick num_vis samples to visualize
    # num_vis: number of samples to visualize

    # the visualization are arranged as follows:
    # first row: y[0, 0, :, :], y[0, t, :, :], y[0, 2*t, :, :],..., y[0, T, :, :]
    # second row: y_pred[0, 0, :, :], y_pred[0, t, :, :], y_pred[0, 2*t, :, :],..., y_pred[0, T, :, :]
    # third row: y[1, 0, :, :], y[1, t, :, :], y[1, 2*t, :, :],..., y[1, T, :, :] and so on

    _, t_total, h, w = y_pred.shape

    dt = t_total // num_t
    fig = plt.figure(figsize=(12, 6))

    y_pred = y_pred[:num_vis, ::dt, :, :]
    y = y[:num_vis, ::dt, :, :]

    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(num_vis*2, num_t),
                     axes_pad=0.05,
                     share_all=False,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="3%",
                     cbar_pad=0.15,
                     )

    # Add data to image grid
    for row in range(num_vis):
        for t in range(num_t):
            grid[row*2*num_t + t].imshow(y_pred[row, t], cmap=cmap)
            grid[row*2*num_t + t].axis('off')
            im = grid[row*2*num_t + t + num_t].imshow(y[row, t], cmap=cmap)
            grid[row*2*num_t + t + num_t].axis('off')
            # grid[row*2*num_t + t + num_t].cax.colorbar(im)
            # grid[row*2*num_t + t + num_t].cax.toggle_label(True)

    # save the figure
    plt.savefig(filename, dpi=200)
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