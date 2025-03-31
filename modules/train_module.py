import lightning as L
import torch
import torch.nn as nn 
from modules.models.dit import ClimaDIT
from dataset.plasim import SURFACE_FEATURES, MULTI_LEVEL_FEATURES
from common.loss import LatitudeWeightedMSE, latitude_weighted_rmse
from common.utils import assemble_grid_params, assemble_input, assemble_scalar_params, disassemble_input, plot_result_2d
from modules.diffusion import SphereLinearScheduler

class TrainModule(L.LightningModule):
    def __init__(self,
                 config,
                 normalizer):
        '''
        Module for training ClimaDiT
        '''

        super().__init__()

        self.mse_criterion=LatitudeWeightedMSE(
                loss_module=nn.MSELoss(reduction='none'),
                with_poles=config.data.with_poles,
                nlat=config.data.nlat,
                nlon=config.data.nlon,
            )
        
        self.loss_module = SphereLinearScheduler(
            num_train_steps=config.training.num_train_steps,
            num_refinement_steps=config.training.num_refinement_steps,
            training_criterion=self.mse_criterion,
            noise_input=config.training.noise_input,
            input_noise_scale=config.training.input_noise_scale,
            l_max=config.training.spherical_l_max,
            noise_type=config.training.noise_type,
            integrator=config.training.integrator,
            restart=config.training.restart,
            restart_step=config.training.restart_step,
        )
        self.normalizer = normalizer
        self.model = ClimaDIT(config)
        self.lr = config.training.lr

    def forward(self, u, sigma_t, scalar_params, grid_params):
        return self.model(u, sigma_t, scalar_params, grid_params)
    
    def training_step(self, batch, batch_idx, eval=False):
        surface_feat, multi_level_feat, constants, yearly_constants, day_of_year, hour_of_day = batch  
        model_input = assemble_input(surface_feat[:, 0], multi_level_feat[:, 0]) # b nlat nlon (c + nlevel*c)
        model_target = assemble_input(surface_feat[:, 1], multi_level_feat[:, 1]) # b nlat nlon (c + nlevel*c)

        scalar_params = assemble_scalar_params(day_of_year, hour_of_day, 0) # b 2
        grid_params = assemble_grid_params(constants, yearly_constants, 0) # b nlat nlon (c + c)
        loss, pred, target = self.loss_module(model_input, model_target, scalar_params, grid_params, self.model)

        if eval:
            return loss, model_input, pred, target

        self.log("train_loss", loss.mean(), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx, eval=False):
        surface_feat, multi_level_feat, constants, yearly_constants, \
            day_of_year, hour_of_day = batch    # did not use cond_param for now

        loss_dict, pred_feat_dict, target_feat_dict = self.predict(
                                                        self.model,
                                                        surface_feat, 
                                                        multi_level_feat,
                                                        day_of_year,
                                                        hour_of_day,
                                                        constants,
                                                        yearly_constants,
                                                        return_pred=True)

        

        # visualize the prediction for first batch
        if batch_idx == 0 and self.config.training.visualize:
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

            plot_result_2d(t2m_pred,
                            t2m_target,
                            f'{self.config.log_dir}/results/val_t2m_{self.current_epoch}.png')
            plot_result_2d(z500_pred,
                            z500_target,
                            f'{self.config.log_dir}/results/val_z500_{self.current_epoch}.png')
        
        if eval:
            return loss_dict, pred_feat_dict, target_feat_dict
        
        # calculate the mean loss
        t2m_loss = torch.cat(loss_dict['tas'], dim=0).mean(0) # surface temp
        z500_loss = torch.cat(loss_dict['zg'], dim=0)[..., 7].mean(0) # geopotential at level=7
        u10m_loss = torch.cat(loss_dict['ua'], dim=0)[..., 0].mean(0) # u wind at level=0
        t850_loss = torch.cat(loss_dict['ta'], dim=0)[..., 10].mean(0) # temp at level=10

        self.log('val_t2m_72', t2m_loss[11].item(), on_step=False, on_epoch=True)
        self.log('val_t2m_120', t2m_loss[19].item(), on_step=False, on_epoch=True)
        self.log('val_t2m_240', t2m_loss[39].item(), on_step=False, on_epoch=True)

        self.log('val_z500_72', z500_loss[11].item(), on_step=False, on_epoch=True)
        self.log('val_z500_120', z500_loss[19].item(), on_step=False, on_epoch=True)
        self.log('val_z500_240', z500_loss[39].item(), on_step=False, on_epoch=True)

        self.log('u10m_72', u10m_loss[11].item(), on_step=False, on_epoch=True)
        self.log('u10m_120', u10m_loss[19].item(), on_step=False, on_epoch=True)
        self.log('u10m_240', u10m_loss[39].item(), on_step=False, on_epoch=True)

        self.log('t850_72', t850_loss[11].item(), on_step=False, on_epoch=True)
        self.log('t850_120', t850_loss[19].item(), on_step=False, on_epoch=True)
        self.log('t850_240', t850_loss[39].item(), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

        return [optimizer], [scheduler]
    

    @torch.no_grad()
    def predict(self, model,
            surface_feat_traj,
            multilevel_feat_traj,
            day_of_year_traj,
            hour_of_day_traj,
            constants_traj,
            yearly_constants_traj,
            return_pred=False # for visualization
            ):
        # surface_feat in shape [b, t, nlat, nlon, num_surface_feats]
        # multilevel_feat in shape [b, t, nlat, nlon, num_levels, num_multilevel_feats]
        # features are normalized

        surface_var_names = SURFACE_FEATURES, 
        multilevel_var_names = MULTI_LEVEL_FEATURES

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
            model_pred = self.loss_module.predict_and_refine(model_input, scalar_params, grid_params, model) # b nlat nlon (c + nlevel*c)
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
        self.normalizer.batch_denormalize(pred_feat_dict)
        self.normalizer.batch_denormalize(target_feat_dict)

        loss_dict = {k:
                        latitude_weighted_rmse(pred_feat_dict[k], target_feat_dict[k],
                                                with_poles=self.config.data.with_poles,
                                                longitude_resolution=self.config.data.nlon,
                                                ) for k in pred_feat_dict.keys()}
        if not return_pred:
            return loss_dict
        else:
            return loss_dict, pred_feat_dict, target_feat_dict