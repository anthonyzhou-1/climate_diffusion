import torch
import torch.nn as nn
from torch_harmonics import InverseRealSHT

# homemade ODE integrator
class ODEIntegrator:
    def __init__(self,
                 method='euler',  # 'euler' or 'heun' or 'midpoint'
                 ):
        self.method = method
        assert method in ['euler', 'heun', 'midpoint'], 'Method not implemented'

    def step_fn(self, x, fn, dt, ts, model_kwargs):
        method = self.method
        if ts[1] == 0 and self.method == 'heun': # prevent irregularity at last time
            method = 'euler'
        if method == 'euler':
            return x + dt * fn(x, ts[0], model_kwargs)
        elif method == 'heun':
            dx = fn(x, ts[0], model_kwargs)
            x1 = x + dt * dx
            return x + 0.5 * dt * (dx + fn(x1, ts[1], model_kwargs))
        elif method == 'midpoint':
            x1 = x + 0.5 * dt * fn(x, ts[0], model_kwargs)
            return x + dt * fn(x1, ts[1], model_kwargs)

    def integrate(self, x, y, model,
                  stencils, timesteps,
                  cond_param=None,
                  grid_param=None,
                  unsqueeze=True):
        if unsqueeze:
            model_wrapper_fn = lambda y, t, model_kwargs: \
                model(torch.cat((x, y.unsqueeze(1)), dim=1), t.expand(x.shape[0]).unsqueeze(-1), **model_kwargs)
        else:
            model_wrapper_fn = lambda y, t, model_kwargs: \
                model(torch.cat((x, y), dim=-1), t.expand(x.shape[0]).unsqueeze(-1), **model_kwargs)

        for i_t in range(len(stencils)-1):
            t_current = stencils[i_t]
            t_next = stencils[i_t+1]
            dt = t_next - t_current
            if self.method != 'midpoint':
                y = self.step_fn(y, model_wrapper_fn, dt,
                                 [timesteps[i_t], timesteps[i_t+1]],
                                 {'scalar_params': cond_param,
                                  'grid_params': grid_param})
            else:
                y = self.step_fn(y, model_wrapper_fn, dt,
                                 [timesteps[i_t], (timesteps[i_t+1] + timesteps[i_t]) / 2],
                                 {'scalar_params': cond_param,
                                  'grid_params': grid_param})
        return y

class DummyScheduler(nn.Module):
    # does nothing but just standard ar training
    def __init__(self,
                 training_criterion=nn.MSELoss(),  # training criterion
                 **kwargs,
                 ):
        super(DummyScheduler, self).__init__()
        self.training_criterion = training_criterion

    def compute_loss(self, x, y, cond_param, grid_param, model):
        y_pred = model(x, None, cond_param, grid_param)
        loss = self.training_criterion(y_pred, y.squeeze(1))
        return loss, y_pred, y

    def forward(self, x, y, cond_param, grid_param, model):
        return self.compute_loss(x, y, cond_param, grid_param, model)

    def predict_and_refine(self, x, cond_param, grid_param, model):
        return model(x, None, cond_param, grid_param)

def unflatten_scalar(x, scalar):
    return scalar.view(-1, *[1 for _ in range(x.ndim - 1)])


class SphereNoiseGenerator(nn.Module):
    def __init__(self, l_max):
        super(SphereNoiseGenerator, self).__init__()
        self.l_max = l_max
        self.isht = InverseRealSHT(l_max, l_max*2, grid="equiangular")

    def forward(self, b, device, dtype=torch.complex64, l_max=None):
        # sample coefficient in the frequency domain
        # b: batch size, l_max: maximum degree
        # return: [b, l_max, l_max + 1] # coefficient for real harmonics
        if l_max is None:
            l_max = self.l_max
            coeffs = torch.randn(b, l_max, l_max + 1, device=device, dtype=dtype)
        else:
            assert l_max <= self.l_max
            coeffs = torch.randn(b, self.l_max, self.l_max + 1, device=device, dtype=dtype)
            # fill with zeros
            coeffs[:, l_max:, :] = 0
        return self.isht(coeffs)

class SphereLinearScheduler(nn.Module):
    def __init__(self,
                 num_refinement_steps,  # this corresponds to physical time steps
                 num_train_steps=None,  # number of training steps
                 training_criterion=nn.MSELoss(),  # training criterion
                 noise_input=False,  # whether to input noise
                 input_noise_scale=1.0,  # the scale of the input noise
                 l_max=8,  # the maximum spherical harmonic degree, if just standard Gaussian this does not matter
                 noise_type='spherical',  # 'spherical' or 'gaussian'
                 integrator='euler',  # 'euler' or 'heun' or 'midpoint', worth noting that this only available for flow
                 restart=False,  # whether to restart the sampling
                 restart_step=None,  # the sigma to restart the sampling
                 ):
        super(SphereLinearScheduler, self).__init__()

        self.input_noise_scale = input_noise_scale

        # for flow matching, the min_noise_std is not used
        self.num_train_timesteps = num_train_steps if num_train_steps is not None else num_refinement_steps + 1
        self.num_refinement_steps = num_refinement_steps
        self.sigmas = torch.linspace(0, 1,
                                     steps=self.num_train_timesteps)

        self.num_refinement_steps = num_refinement_steps
        self.ode_integrator = ODEIntegrator(method=integrator)

        self.restart = restart
        self.restart_step = restart_step

        self.training_criterion = training_criterion
        self.noise_input = noise_input

        self.noise_type = noise_type
        if noise_type == 'spherical':
            # currently the max grid resolution is hard coded
            self.noise_generator = SphereNoiseGenerator(128)
            self.l_max = l_max
        else:
            self.noise_generator = None
            self.l_max = None

    def get_noise(self, size, device):
        if self.noise_type == 'spherical':
            b = size[0]
            return self.noise_generator(b, device, l_max=self.l_max)
        else:
            return torch.randn(size, device=device)

    def compute_loss(self, x, y, cond, grid_cond, model):
        # for now, assume it's Markovian
        # x: [b nlat nlon d]
        # y: [b nlat nlon d]
        # cond: [b 2] in this case
        # grid_cond: [b nlat nlon c]
        noise = self.get_noise(size=y.shape, device=y.device).to(y.dtype)

        # no need to train on k=0
        k = torch.randint(1, self.num_train_timesteps, device=x.device, size=(x.shape[0],)).long()

        # retrieve from the scheduler

        sigma_t = self.sigmas.to(x.device)[k]
        alpha_t = (1 - sigma_t)
        alpha_t = alpha_t.view(-1, *[1 for _ in range(y.ndim - 1)])
        sigma_t = sigma_t.view(-1, *[1 for _ in range(y.ndim - 1)])
        y_noised = alpha_t * y + sigma_t * noise

        if self.noise_input:
            # do the reverse of the y noising scheme
            x = x + alpha_t * torch.randn_like(x) * self.input_noise_scale

        y_noised = y_noised
        u_in = torch.cat([x, y_noised], dim=-1)  # input both condition and noised prediction, [b nlat nlon 2d]
        pred = model(u_in, k.float().view(-1, 1), cond, grid_cond) # the cond is in range [0, 1]
        target = noise - y
        loss = self.training_criterion(pred, target)
        return loss, pred, target

    def predict_and_refine(self, x, cond_param, grid_param,
                           model, refinement_steps=None):
        if refinement_steps is None:
            refinement_steps = self.num_refinement_steps

        # x: [b c h w]
        y_noised = self.get_noise(
            x.shape, device=x.device
        ).to(x.dtype)

        timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1, device=x.device).long()
        # trailing timesteps
        timesteps = timesteps[::((self.num_train_timesteps - 1) // refinement_steps)]
        sigmas = self.sigmas.to(x.device)[timesteps]

        # currently does not support noising input
        integrator = self.ode_integrator
        y_noised = integrator.integrate(x, y_noised, model, sigmas, timesteps, cond_param, grid_param,
                                            unsqueeze=False)

        if self.restart:  # more like sde-edit
            sigmas = sigmas[self.restart_step:]
            timesteps = timesteps[self.restart_step:]
            y_noised = y_noised * (1 - sigmas[0]) + torch.randn_like(y_noised) * sigmas[0]
            y_noised = integrator.integrate(x, y_noised, model, sigmas, timesteps, cond_param, grid_param,
                                            unsqueeze=False)

        y = y_noised
        return y

    def forward(self, x, y, scalar_params, grid_params, model):
        return self.compute_loss(x, y, scalar_params, grid_params, model)
