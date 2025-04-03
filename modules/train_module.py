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
        self.config = config

        self.ddp = True if config.strategy == 'ddp' else False

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

        self.log("train_loss", loss.mean(), on_step=True, on_epoch=True, sync_dist=self.ddp)

        return loss

    def validation_step(self, batch, batch_idx, eval=False):
        surface_feat, multi_level_feat, constants, yearly_constants, \
            day_of_year, hour_of_day = batch    

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
        if batch_idx == 0 and self.config.training.visualize and self.global_rank == 0:
            t2m_pred = pred_feat_dict['tas'][0].cpu().numpy()
            t2m_target = target_feat_dict['tas'][0].cpu().numpy()
            z500_pred = pred_feat_dict['zg'][0, ..., 7].cpu().numpy()
            z500_target = target_feat_dict['zg'][0, ..., 7].cpu().numpy()

            plot_result_2d(t2m_pred, # t h w
                            t2m_target,
                            f'{self.config.log_dir}/val_t2m_{self.current_epoch}.png')
            plot_result_2d(z500_pred,
                            z500_target,
                            f'{self.config.log_dir}/val_z500_{self.current_epoch}.png')
        
        if eval:
            return loss_dict, pred_feat_dict, target_feat_dict
        
        # calculate the mean loss, shape b t for each key, b t l for multilevel keys
        t2m_loss = loss_dict['tas'].mean(0) # surface temp, mean across batch dim
        pr_6h_loss = loss_dict['pr_6h'].mean(0) # 6-hour accumulated precipitation
        z500_loss = loss_dict['zg'][..., 7].mean(0) # geopotential at level=7
        u250_loss = loss_dict['ua'][..., 4].mean(0) # u wind at level=4
        t850_loss = loss_dict['ta'][..., 10].mean(0) # temp at level=10
        
        self.log('val/t2m_6', t2m_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 6 hours
        self.log('val/t2m_24', t2m_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 1 day
        self.log('val/t2m_72', t2m_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 3 day
        self.log('val/t2m_120', t2m_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 5 day
        self.log('val/t2m_240', t2m_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 10 day

        self.log('val/pr_6h_6', pr_6h_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/pr_6h_24', pr_6h_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/pr_6h_72', pr_6h_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/pr_6h_120', pr_6h_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/pr_6h_240', pr_6h_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

        self.log('val/z500_6', z500_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500_24', z500_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500_72', z500_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500_120', z500_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500_240', z500_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

        self.log('val/u250_6', u250_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250_24', u250_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250_72', u250_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250_120', u250_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250_240', u250_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

        self.log('val/t850_6', t850_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850_24', t850_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850_72', t850_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850_120', t850_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850_240', t850_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

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

        surface_var_names = SURFACE_FEATURES 
        multilevel_var_names = MULTI_LEVEL_FEATURES

        surface_init = surface_feat_traj[:, 0] # b nlat nlon c
        multilevel_init = multilevel_feat_traj[:, 0] # b nlat nlon nlevel c
        model_input = assemble_input(surface_init, multilevel_init) # b nlat nlon (c + nlevel*c)

        surface_target = surface_feat_traj[:, 1:] # b t nlat nlon c
        multilevel_target = multilevel_feat_traj[:, 1:] # b t nlat nlon nlevel c

        surface_pred = torch.zeros_like(surface_target, device=surface_init.device) # b t nlat nlon c
        multilevel_pred = torch.zeros_like(multilevel_target, device=multilevel_init.device) # b t nlat nlon nlevel c

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

        surface_pred, multilevel_pred = self.normalizer.batch_denormalize(surface_pred, multilevel_pred)

        pred_feat_dict = {}
        target_feat_dict = {}
        for c, surface_feat_name in enumerate(surface_var_names):
            pred_feat_dict[surface_feat_name] = surface_pred[..., c]
            target_feat_dict[surface_feat_name] = surface_target[..., c]

        for c, multilevel_feat_name in enumerate(multilevel_var_names):
            pred_feat_dict[multilevel_feat_name] = multilevel_pred[..., c]
            target_feat_dict[multilevel_feat_name] = multilevel_target[..., c]

        loss_dict = {k:
                        latitude_weighted_rmse(pred_feat_dict[k], target_feat_dict[k],
                                                with_poles=self.config.data.with_poles,
                                                longitude_resolution=self.config.data.nlon,
                                                ) for k in pred_feat_dict.keys()} # b t for each key
        if not return_pred:
            return loss_dict
        else:
            return loss_dict, pred_feat_dict, target_feat_dict
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)

        return [optimizer], [scheduler]