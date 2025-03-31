import argparse
import torch
import os, sys
import logging
import yaml, shutil
from einops import rearrange

from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt


class LogBuffer:

    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False

    def clear(self) -> None:
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self) -> None:
        self.output.clear()
        self.ready = False

    def update(self, vars: dict, count: int = 1) -> None:
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(var)
            self.n_history[key].append(count)

    def average(self, n: int = 0) -> None:
        """Average latest n values or all values."""
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
        self.ready = True



import matplotlib
from mpl_toolkits.axes_grid1 import ImageGrid


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


def log(logger, logging_info, print_info=True):
    logger.info(logging_info)
    if print_info:
        print(logging_info)


def prepare_training(args, config):

    log_dir = config.log_dir

    # prepare the logger
    # ensure the directory to save the model
    # first check if the log directory exists
    os.umask(0o000)

    ensure_dir(log_dir)
    ensure_dir(log_dir + '/model')
    ensure_dir(log_dir + '/code_cache')
    ensure_dir(log_dir + '/results')

    logger = logging.getLogger("LOG")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, 'logging_info'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # copy config yaml file to log_dir
    shutil.copyfile(args.config, os.path.join(log_dir, 'config.yml'))
    # copy all the code to code_cache folder, including current training script
    # copy the script itself
    shutil.copyfile(sys.argv[0], os.path.join(log_dir, 'code_cache', sys.argv[0]))
    shutil.copytree('models/', os.path.join(log_dir, 'code_cache', 'libs'), dirs_exist_ok=True)
    shutil.copytree('datasets/', os.path.join(log_dir, 'code_cache', 'dataset'), dirs_exist_ok=True)
    shutil.copyfile('training_utils.py', os.path.join(log_dir, 'code_cache', 'training_utils.py'))

    return logger


def dump_state(model, optimizer, scheduler, global_step, log_dir, ema=None):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'global_step': global_step,
    }
    if ema is not None:
        state_dict['ema'] = ema.state_dict()

    save_path = log_dir + '/model' + f'/model_{(global_step // 1000)}k_iter.pth'
    torch.save(state_dict, save_path)


def load_state(model, checkpoint,
               optim=None, sched=None, ema=None,
               resume_training_state=True,
               logger=None):
    missing, unexpected = model.load_state_dict(checkpoint['model'])
    if logger is not None:
        logger.info(f'Missing keys: {missing}')
        logger.info(f'Unexpected keys: {unexpected}')
    if resume_training_state:
        assert optim is not None and sched is not None, 'Optimizer and scheduler must be provided when resuming training state'
        optim.load_state_dict(checkpoint['optimizer'])
        sched.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']
        if 'ema' in checkpoint and ema is not None:
            ema.load_state_dict(checkpoint['ema'])
        return global_step
    else:
        return 0    # start from scratch


def dump_multiple_state(model_lst, optimizer, scheduler, global_step, log_dir, ema_lst=None):
    state_dict = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'global_step': global_step,
    }
    for i, model in enumerate(model_lst):
        state_dict[f'model_{i}'] = model.state_dict()
        if ema_lst is not None:
            state_dict[f'ema_{i}'] = ema_lst[i].state_dict()

    save_path = log_dir + '/model' + f'/model_{(global_step // 1000)}k_iter.pth'
    torch.save(state_dict, save_path)


def load_multiple_state(model_lst, checkpoint,
                        optim=None, sched=None, ema_lst=None,
                        resume_training_state=True,
                        logger=None):
    for i, model in enumerate(model_lst):
        missing, unexpected = model.load_state_dict(checkpoint[f'model_{i}'])
        if logger is not None:
            logger.info(f'Loaded model {i}')
            logger.info(f'Missing keys: {missing}')
            logger.info(f'Unexpected keys: {unexpected}')
        if 'ema' in checkpoint and ema_lst is not None:
            ema_lst[i].load_state_dict(checkpoint[f'ema_{i}'])

    if resume_training_state:
        assert optim is not None and sched is not None, 'Optimizer and scheduler must be provided when resuming training state'
        optim.load_state_dict(checkpoint['optimizer'])
        sched.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']

        return global_step
    else:
        return 0    # start from scratch


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    try:
        return data.to(device, non_blocking=True)
    except:
        return data


@torch.no_grad()
def ema_update(model, ema_model, decay):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data = decay * ema_param.data + (1 - decay) * (param.data).detach()


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--comment', type=str, default='', help='Comment')
    parser.add_argument('--global_seed', type=int, default=970314, help='Global seed')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    args = parser.parse_args()

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    # copy the config file to the log_dir
    return args, config