# Basic imports
import gc
import os
import time
import datetime
from copy import deepcopy

# Library imports
import accelerate
import dask
import torch
import torch.nn as nn
from accelerate import Accelerator, InitProcessGroupKwargs
from torch.utils.data import DataLoader
from transformers import get_scheduler

# Custom imports
from training_utils import parse_args_and_config, log, prepare_training, load_state, dump_state, to_device, assemble_grid_params, assemble_input, assemble_scalar_params, disassemble_input, count_params
from logging_utils import LogBuffer, plot_result_2d
from losses.weather_loss_fn import latitude_weighted_rmse, LatitudeWeightedMSE
from losses.spherical_diffusion import SphereLinearScheduler
from datasets.plasim import PLASIMData, SURFACE_FEATURES, MULTI_LEVEL_FEATURES, CONSTANTS_FEATURES, YEARLY_FEATURES

# Setup
dask.config.set(scheduler='synchronous')
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def configure_model(config):
    if config.model.name == 'dit':
        from models.dit import ClimaDIT
        model = ClimaDIT(config)
        print(f"Number of parameters: {count_params(model)}")
    else:
        raise NotImplementedError(f"Model {config.model.name} not implemented")
    if config.training.compile_model:
        model = torch.compile(model)
    return model

def configure_optimizer(config):
    # configure the optimizer and scheduler
    # note: no weight decay used
    optimizer = torch.optim.AdamW(model.parameters(),
                      lr=config.training.lr, # this is the constant lr
                      betas=(config.training.beta1, config.training.beta2))

    scheduler = get_scheduler('cosine_with_min_lr',
                              optimizer,
                              num_warmup_steps=config.training.warmup_steps*accelerator.num_processes,
                                num_training_steps=config.training.total_steps*accelerator.num_processes,
                              scheduler_specific_kwargs={'min_lr': config.training.min_lr})

    return optimizer, scheduler


def configure_dataloader_and_normalizer(config, batch_size):
    # only get valid dataloader on rank 0

    train_data = PLASIMData(data_path=config.data.train_data_path,
                            norm_stats_path=config.data.norm_stats_path,
                            boundary_path=config.data.boundary_path,
                            surface_vars=SURFACE_FEATURES,
                            multi_level_vars=MULTI_LEVEL_FEATURES,
                            constant_names=CONSTANTS_FEATURES,
                            yearly_names=YEARLY_FEATURES,
                            nsteps=config.data.training_nsteps,     
                             )
    # build dataloader
    train_loader = DataLoader(
                            dataset=train_data,
                            num_workers=config.training.num_workers,
                            batch_size=batch_size,
                            drop_last=True,
                            shuffle=True,
                            prefetch_factor=3,
                            persistent_workers=True,
                            pin_memory=True
        )

    val_dataset = PLASIMData(data_path=config.data.val_data_path,
                            norm_stats_path=config.data.norm_stats_path,
                            boundary_path=config.data.boundary_path,
                            surface_vars=SURFACE_FEATURES,
                            multi_level_vars=MULTI_LEVEL_FEATURES,
                            constant_names=CONSTANTS_FEATURES,
                            yearly_names=YEARLY_FEATURES,
                            nsteps=config.data.val_nsteps,
                            load_into_memory=True)
    
    
    surface_vars, multi_level_vars, constants, yearly_constants = val_dataset.get_var_names()
    data_normalizer = val_dataset.normalizer

    val_dataloader = DataLoader(
        dataset=val_dataset,
        num_workers=4,
        batch_size=config.training.eval_batch_size,
        drop_last=False,
        shuffle=False,
    )

    return train_loader, val_dataloader, data_normalizer,\
            surface_vars, multi_level_vars, constants, yearly_constants

@torch.no_grad()
def ema_update(model, ema_model, decay):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data = decay * ema_param.data + (1 - decay) * (param.data).detach()

@torch.no_grad()
def predict(model,
            surface_feat_traj,
            multilevel_feat_traj,
            day_of_year_traj,
            hour_of_day_traj,
            constants_traj,
            yearly_constants_traj,
            surface_var_names,  # these are used to create a dict for the losses
            multilevel_var_names,
            return_pred=False # for visualization
            ):
    # surface_feat in shape [b, t, nlat, nlon, num_surface_feats]
    # multilevel_feat in shape [b, t, nlat, nlon, num_levels, num_multilevel_feats]
    # features are normalized

    # please make sure the model is in eval mode
    # please put all the tensor on correct device before calling this function

    surface_init = surface_feat_traj[:, 0] # b nlat nlon c
    multilevel_init = multilevel_feat_traj[:, 0] # b nlat nlon nlevel c
    model_input = assemble_input(surface_init, multilevel_init) # b nlat nlon (c + nlevel*c)

    surface_target = surface_feat_traj[:, 1:] # b t nlat nlon c
    multilevel_target = multilevel_feat_traj[:, 1:] # b t nlat nlon nlevel c

    surface_pred = torch.zeros_like(surface_target) # b t nlat nlon c
    multilevel_pred = torch.zeros_like(multilevel_target) # b t nlat nlon nlevel c

    # let's predict!

    for t in range(surface_target.shape[1]):
        # assemble conditional info
        scalar_params = assemble_scalar_params(day_of_year_traj, hour_of_day_traj, t) # b, 2
        grid_params = assemble_grid_params(constants_traj, yearly_constants_traj, t) # b nlat nlon (c + t*c)
        # make prediction
        model_pred = diffusion_loss_module.predict_and_refine(model_input, scalar_params, grid_params, model) # b nlat nlon (c + nlevel*c)
        # rearrange prediction and save
        surface_pred_t, multilevel_pred_t = disassemble_input(model_pred, num_levels=multilevel_init.shape[-2], num_surface_channels=surface_init.shape[-1])
        surface_pred[:, t] = surface_pred_t
        multilevel_pred[:, t] =  multilevel_pred_t
        # update model_input
        model_input = model_pred

    pred_feat_dict = {}
    target_feat_dict = {}
    for c, surface_feat_name in enumerate(surface_var_names):
        pred_feat_dict[surface_feat_name] = surface_pred[..., c]
        target_feat_dict[surface_feat_name] = surface_target[..., c]

    for c, multilevel_feat_name in enumerate(multilevel_var_names):
        pred_feat_dict[multilevel_feat_name] = multilevel_pred[..., c]
        target_feat_dict[multilevel_feat_name] = multilevel_target[..., c]

    # calculate the unnormalized loss
    normalizer.batch_denormalize(pred_feat_dict)
    normalizer.batch_denormalize(target_feat_dict)

    loss_dict = {k:
                     latitude_weighted_rmse(pred_feat_dict[k], target_feat_dict[k],
                                            with_poles=config.data.with_poles,
                                            longitude_resolution=config.data.nlon,
                                            ) for k in pred_feat_dict.keys()}
    if not return_pred:
        return loss_dict
    else:
        return loss_dict, pred_feat_dict, target_feat_dict

def validate_loop():
    if accelerator.is_main_process:
        log(logger, '====================================')
        log(logger, f'Validating at step: {global_step}...')

        # print how many sampling steps
        if not hasattr(config.training, 'ar_training'):
            log(logger, f'Validating with {config.training.num_refinement_steps} sampling steps')

    ema_model_temp = deepcopy(accelerator.unwrap_model(model))
    ema_model_temp.requires_grad_(False).eval()

    # load from ckpt's ema model
    ema_model_state_dict = torch.load(os.path.join(log_dir, 'ema_latest.pth'))
    if accelerator.is_main_process:
        log(logger, 'Loaded EMA model from ckpt...')
    ema_model_temp.load_state_dict(ema_model_state_dict)
    ema_model_temp.to(accelerator.device)
    ema_model_temp.eval()

    loss_dict_all = {}
    # randomly select a batch for main_process to visualize
    i_vis = 0
    i = 0
    data_iter = iter(val_loader)
    data_len = len(val_loader)
    if accelerator.is_main_process:
        log(logger, f'len of val_loader: {data_len}')
    while True:
        try:
            batch = next(data_iter)
        except StopIteration:
            if accelerator.is_main_process:
                log(logger, 'Finish validation')
            break
        batch = to_device(batch, accelerator.device)
        surface_feat, multi_level_feat, constants, yearly_constants, \
            day_of_year, hour_of_day = batch    # did not use cond_param for now

        loss_dict, pred_feat_dict, target_feat_dict = predict(
                                                        ema_model_temp,
                                                    surface_feat, 
                                                    multi_level_feat,
                                                    day_of_year,
                                                    hour_of_day,
                                                    constants,
                                                    yearly_constants,
                                                    surface_vars,
                                                    multi_level_vars,
                                                    return_pred=True)

        loss_dict_lst = [loss_dict]
        # gather the loss from all processes
        loss_dict_lst = accelerate.utils.gather_object(loss_dict_lst)

        if accelerator.is_main_process:
            # concat all the gathered dict
            loss_dict = {
                k: torch.cat([d[k].to(accelerator.device) for d in loss_dict_lst], dim=0)
                for k in loss_dict_lst[0].keys()
            }

            # concat to loss_dict_all
            for k in loss_dict.keys():
                if k not in loss_dict_all:
                    loss_dict_all[k] = []
                loss_dict_all[k].append(loss_dict[k])

            # visualize the prediction
            if i == i_vis and config.training.visualize:
                os.umask(0o000)
                t2m_pred = pred_feat_dict['tas'].cpu().numpy()
                t2m_target = target_feat_dict['tas'].cpu().numpy()
                z500_pred = pred_feat_dict['zg'][..., 7].cpu().numpy()
                z500_target = target_feat_dict['zg'][..., 7].cpu().numpy()
                # b t h w
                # we just need 4 batches of each
                t2m_pred = t2m_pred[:4]
                t2m_target = t2m_target[:4]
                z500_pred = z500_pred[:4]
                z500_target = z500_target[:4]

                # reshape to [b, t, h, w]
                t2m_pred = t2m_pred.reshape(4, -1, t2m_pred.shape[-2], t2m_pred.shape[-1])
                t2m_target = t2m_target.reshape(4, -1, t2m_target.shape[-2], t2m_target.shape[-1])
                z500_pred = z500_pred.reshape(4, -1, z500_pred.shape[-2], z500_pred.shape[-1])
                z500_target = z500_target.reshape(4, -1, z500_target.shape[-2], z500_target.shape[-1])
                # t850_pred = t850_pred.reshape(4, -1, t850_pred.shape[-2], t850_pred.shape[-1])
                # t850_target = t850_target.reshape(4, -1, t850_target.shape[-2], t850_target.shape[-1])

                plot_result_2d(t2m_pred,
                               t2m_target,
                               f'{config.log_dir}/results/val_t2m_{global_step}.png')
                plot_result_2d(z500_pred,
                               z500_target,
                               f'{config.log_dir}/results/val_z500_{global_step}.png')
                # plot_result_2d(t850_pred,
                #                t850_target,
                #                f'{config.log_dir}/results/val_t850_{global_step}.png')
        accelerator.wait_for_everyone()
        i += 1

    if accelerator.is_main_process:
        # currently a bunch of interested variables are hard-coded
        # 2m_temperature, 10m_u_component_of_wind, 10m_v_component_of_wind
        # geopotential at 500hPa, temperature at 850hPa

        # the time step we report error is also hard coded at
        # 24hr, 72hr, 120hr, 168hr, 240hr
        # corresponds to 4step, 12step, 20step, 28step, 40step

        # calculate the mean loss
        t2m_loss = torch.cat(loss_dict_all['tas'], dim=0).mean(0)
        z500_loss = torch.cat(loss_dict_all['zg'], dim=0)[..., 7].mean(0)
        u10m_loss = torch.cat(loss_dict_all['ua'], dim=0)[..., 0].mean(0)
        t850_loss = torch.cat(loss_dict_all['ta'], dim=0)[..., 10].mean(0)

        log(logger, f'Validation Prediction Loss on 2m temperature 24/72/120/168/240hr:'
                    f' {t2m_loss[3].item():.4f},'
                    f' {t2m_loss[11].item():.4f},'
                    f' {t2m_loss[19].item():.4f},'
                    f' {t2m_loss[27].item():.4f},'
                    f' {t2m_loss[39].item():.4f}')
        log(logger, f'Validation Prediction Loss on z500 24/72/120/168/240hr:'
                    f' {z500_loss[3].item():.1f},'
                    f' {z500_loss[11].item():.1f},'
                    f' {z500_loss[19].item():.1f},'
                    f' {z500_loss[27].item():.1f},'
                    f' {z500_loss[39].item():.1f}')
        log(logger, f'Validation Prediction Loss on u10m 24/72/120/168/240hr:'
                    f' {u10m_loss[3].item():.4f},'
                    f' {u10m_loss[11].item():.4f},'
                    f' {u10m_loss[19].item():.4f},'
                    f' {u10m_loss[27].item():.4f},'
                    f' {u10m_loss[39].item():.4f}')
        log(logger, f'Validation Prediction Loss on t850 24/72/120/168/240hr:'
                    f' {t850_loss[3].item():.4f},'
                    f' {t850_loss[11].item():.4f},'
                    f' {t850_loss[19].item():.4f},'
                    f' {t850_loss[27].item():.4f},'
                    f' {t850_loss[39].item():.4f}')

        # for logging log 72 120 and 240
        accelerator.log({
            'val_t2m_72': t2m_loss[11].item(),
            'val_t2m_120': t2m_loss[19].item(),
            'val_t2m_240': t2m_loss[39].item(),

            'val_z500_72': z500_loss[11].item(),
            'val_z500_120': z500_loss[19].item(),
            'val_z500_240': z500_loss[39].item(),

            'val_u10m_72': u10m_loss[11].item(),
            'val_u10m_120': u10m_loss[19].item(),
            'val_u10m_240': u10m_loss[39].item(),

            'val_t850_72': t850_loss[11].item(),
            'val_t850_120': t850_loss[19].item(),
            'val_t850_240': t850_loss[39].item(),
        }, step=global_step)

        log(logger, '====================================')
        # clear cuda cache
        del loss_dict_all

    del ema_model_temp
    torch.cuda.empty_cache()
    gc.collect()

    return


def train():
    global global_step

    training_iter = iter(train_loader)
    log_buffer = LogBuffer()

    # skip some data if necessary
    if skip_step > 0:
        for _ in range(skip_step):
            try:
                _ = next(training_iter)
            except StopIteration:
                training_iter = iter(train_loader)
                _ = next(training_iter)

    data_time_start = time.time()
    data_time_all = 0
    time_start, last_tic = time.time(), time.time()

    while global_step < config.training.total_steps:
        try:
            batch = next(training_iter)
        except StopIteration:
            training_iter = iter(train_loader)
            batch = next(training_iter)

        # retrieve things from batch
        batch = to_device(batch, accelerator.device)
        surface_feat, multi_level_feat, constants, yearly_constants, \
            day_of_year, hour_of_day = batch  

        model_input = assemble_input(surface_feat[:, 0], multi_level_feat[:, 0]) # b nlat nlon (c + nlevel*c)
        model_target = assemble_input(surface_feat[:, 1], multi_level_feat[:, 1]) # b nlat nlon (c + nlevel*c)

        scalar_params = assemble_scalar_params(day_of_year, hour_of_day, 0) # b 2
        grid_params = assemble_grid_params(constants, yearly_constants, 0) # b nlat nlon (c + c)

        data_time_all += time.time() - data_time_start
        with accelerator.accumulate(model):
            optimizer.zero_grad()

            loss, _, _ = diffusion_loss_module(
                model_input, model_target, scalar_params, grid_params, model
            )

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()
            scheduler.step()

        lr = scheduler.get_last_lr()[0]
        logs = {'loss': accelerator.gather(loss).mean().item()}

        if grad_norm is not None:
            logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
        log_buffer.update(logs)

        if (global_step + 1) % config.log_interval == 0 or (global_step + 1) == 1:
            print(f"Loss: {logs['loss']}")
            t = (time.time() - last_tic) / config.log_interval
            t_d = data_time_all / config.log_interval
            avg_time = (time.time() - time_start) / (global_step + 1)
            eta = str(datetime.timedelta(
                seconds=int(avg_time * (config.training.total_steps - global_step - 1))))
            log_buffer.average()

            info = f"Step [{global_step}/{config.training.total_steps}]]:" \
                   f"total_eta: {eta}, " \
                   f"time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, "

            info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
            if accelerator.is_main_process:
                log(logger, info)
            log_buffer.clear()

            last_tic = time.time()
            data_time_all = 0

        logs.update(lr=lr)
        accelerator.log(logs, step=global_step)

        # do an ema update
        if accelerator.is_main_process:
            if global_step < config.training.ema_after:
                pass
            elif global_step == config.training.ema_after:
                # copy param
                ema_model.load_state_dict(model.state_dict())
            else:
                ema_update(model, ema_model, config.training.ema_decay)

        global_step += 1
        data_time_start = time.time()

        if global_step % config.save_model_steps == 0 or \
                global_step == 1:
            if accelerator.is_main_process:
                os.umask(0o000)
                dump_state(accelerator.unwrap_model(model),
                           optimizer,
                           scheduler,
                           global_step,
                           log_dir,
                           ema=ema_model)

            accelerator.wait_for_everyone()

        if global_step % config.eval_sampling_steps == 0 or \
            global_step == 1:

            if global_step <= config.training.ema_after and accelerator.is_main_process:
                ema_update(model, ema_model, 0.0)

                # save the ema model to a temporary file
            if accelerator.is_main_process:
                torch.save(ema_model.state_dict(), f'{log_dir}/ema_latest.pth')
            accelerator.wait_for_everyone()

            validate_loop()

    if accelerator.is_main_process:
        os.umask(0o000)
        dump_state(accelerator.unwrap_model(model),
                   optimizer,
                   scheduler,
                   global_step,
                   log_dir,
                   ema=ema_model)
        torch.save(ema_model.state_dict(), f'{log_dir}/ema_latest.pth')
    accelerator.wait_for_everyone()

    validate_loop()
    if accelerator.is_main_process:
        log(logger, 'Training finished...')
    accelerator.end_training()
    exit()


if __name__ == "__main__":

    args, config = parse_args_and_config()
    # prepare the training
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=3600)  # change timeout to avoid a strange NCCL bug

    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f'{config.model.name}_{now}'
    config.log_dir = config.log_dir + '/' + run_name

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        project_dir=os.path.join(config.log_dir, "logs"),
        log_with='wandb',
        kwargs_handlers=[init_handler]
    )
    accelerator.init_trackers(
        project_name=config.project_name,
        config=config,
        init_kwargs={"wandb":{"name":run_name}}
    )

    if accelerator.is_main_process:
        logger = prepare_training(args, config)
    else:
        logger = None

    log_dir = config.log_dir

    # configure the model
    model = configure_model(config)
    model.train()

    # configure the optimizer
    optimizer, scheduler = configure_optimizer(config)

    # configure dataloader and normalizer
    train_loader, val_loader, normalizer,\
        surface_vars, multi_level_vars, constants_names, yearly_constants_names  = \
        configure_dataloader_and_normalizer(config,
                                            config.training.batch_size_per_device)

    global_step = 0
    skip_step = config.training.skip_step

    # optionally resume training:
    if config.training.resume:
        ckpt = torch.load(config.training.resume_from, map_location='cpu')
        global_step = load_state(
                            model,
                            ckpt,
                            optimizer,
                            scheduler,
                            resume_training_state=config.training.resume_training_state,
                            logger=logger
                        )
        if accelerator.is_main_process:
            log(logger, f"Resumed training from {global_step}...")

    if accelerator.is_main_process:
        ema_model = deepcopy(model)
        ema_model.requires_grad_(False)
        ema_model.eval()
        ema_model.to(accelerator.device)
        log(logger, 'EMA model created successfully')
    model.to(accelerator.device)

    # prepare for distributed training
    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)


    if hasattr(config.training, 'ar_training'):
        from losses.diffusion_scheduler import DummyScheduler
        diffusion_loss_module = DummyScheduler(
            training_criterion=LatitudeWeightedMSE(
                loss_module=nn.MSELoss(reduction='none'),
                with_poles=config.data.with_poles,
                nlat=config.data.nlat//config.model.patch_size,
                nlon=config.data.nlon//config.model.patch_size,
            ),)
    else:
        diffusion_loss_module = SphereLinearScheduler(
            num_train_steps=config.training.num_train_steps,
            num_refinement_steps=config.training.num_refinement_steps,
            training_criterion=LatitudeWeightedMSE(
                loss_module=nn.MSELoss(reduction='none'),
                with_poles=config.data.with_poles,
                nlat=config.data.nlat,
                nlon=config.data.nlon,
            ),
            noise_input=config.training.noise_input,
            input_noise_scale=config.training.input_noise_scale,
            l_max=config.training.spherical_l_max,
            noise_type=config.training.noise_type,
            integrator=config.training.integrator,
            restart=config.training.restart,
            restart_step=config.training.restart_step,
        )
    diffusion_loss_module.to(accelerator.device)
    train()




