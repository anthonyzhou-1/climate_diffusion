import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange

from modules.models.positional_encoding import TimestepEmbedder

from modules.models.spherical_harmonics import SphericalHarmonicsPE

from modules.models.basics import MLP, LayerNorm, \
    bias_dropout_add_scale_fused_train, \
    bias_dropout_add_scale_fused_inference, \
    modulate_fused

from modules.models.factorized_attention import FADiTBlockS2

#################################################################################
#                                 Core Model                                    #
#################################################################################
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
     
class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout = 0.0,
            act='gelu',
            mlp_ratio=4,
    ):
        super().__init__()
        self.ln_q = nn.LayerNorm(hidden_dim)
        self.ln_kv = nn.LayerNorm(hidden_dim)
        self.Attn = CrossAttention(hidden_dim, hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                dropout=dropout) # assume query and context dim are the same
            
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, expansion_ratio=mlp_ratio)

    def forward(self, q, kv):
        fx = self.Attn(self.ln_q(q), self.ln_kv(kv)) + q
        fx = self.mlp(self.ln_2(fx)) + fx

        return fx

class DiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim,
                       expansion_ratio=mlp_ratio)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x,
                scalar_cond):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa, shift_mlp,
         scale_mlp, gate_mlp) = self.adaLN_modulation(scalar_cond)[:, None].chunk(6, dim=2)

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv,
                        'b s (three h d) -> b h three s d',
                        three=3,
                        h=self.n_heads)
        qk, v = qkv[:, :, :2], qkv[:, :, 2]
        # print(qk.shape)

        q, k = qk[:, :, 0], qk[:, :, 1]
        # use F.scale dot product attention
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)

        x = rearrange(x, 'b h s d -> b s (h d)', b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x),
                                  None,
                                  gate_msa,
                                  x_skip,
                                  self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(
                self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout)
        return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size=2,
            in_chans=3,
            hidden_size=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        embed_dim = hidden_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_params(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        x = rearrange(x, 'b ny nx c -> b c ny nx')
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        else:
            x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.norm(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size,
                 cond_dim,
                 patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class ClimaDIT(nn.Module):
    # didn't use rope in this model
    #
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.in_dim = config.model.in_dim
        self.out_dim = config.model.out_dim
        self.dim = config.model.dim
        self.cond_dim = config.model.cond_dim
        self.num_heads = config.model.num_heads
        self.num_fa_blocks = config.model.num_fa_blocks
        self.num_sa_blocks = config.model.num_sa_blocks
        self.num_ca_blocks = config.model.num_ca_blocks
        self.num_cond = config.model.num_cond
        self.patch_size = config.model.patch_size
        self.num_constants = config.model.num_constants
        self.nlat, self.nlon = config.data.nlat, config.data.nlon

        self.grid_x = self.nlat // self.patch_size
        self.grid_y = self.nlon // self.patch_size

        self.with_poles = config.data.with_poles

        # grid embedding
        self.constant_embedder = nn.Sequential(
            Rearrange('b ny nx c -> b c ny nx'),
            nn.Conv2d(self.num_constants,
                      self.dim,
                      kernel_size=self.patch_size, stride=self.patch_size, padding=0),
            nn.SiLU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0),
            Rearrange('b c ny nx -> b ny nx c')
        )

        # input embedding
        self.patch_embed = PatchEmbed(patch_size=self.patch_size,
                                      in_chans=self.in_dim,
                                      hidden_size=self.dim,
                                      flatten=False)

        # positional embedding
        self.pe_embed = SphericalHarmonicsPE(config.model.l_max, self.dim, self.dim,
                                             use_mlp=True)
        self.pe2patch = PatchEmbed(patch_size=self.patch_size,
                                   in_chans=self.dim,
                                   hidden_size=self.dim,
                                   flatten=False)

        # scalar embedding
        if self.num_cond > 0:
            self.cond_map = TimestepEmbedder(self.cond_dim, num_conds=self.num_cond)

        fa_blocks = []
        for _ in range(self.num_fa_blocks):
            fa_blocks.append(FADiTBlockS2(self.dim,
                                       self.dim // self.num_heads,
                                       self.num_heads,
                                       config.model.proj_bottleneck_dim,
                                       self.dim,
                                       self.cond_dim,
                                       kernel_expansion_ratio=config.model.kernel_expansion_ratio,
                                       use_softmax=True,
                                       depth_dropout=config.model.depth_dropout))

        sa_blocks = []
        for _ in range(self.num_sa_blocks):
            sa_blocks.append(DiTBlock(self.dim,
                                     self.num_heads,
                                     self.cond_dim,
                                     dropout=config.model.depth_dropout))
            
        ca_blocks = []
        for _ in range(self.num_ca_blocks):
            ca_blocks.append(CrossAttentionBlock(self.num_heads,
                                                 self.dim,))
        self.fa_blocks = nn.ModuleList(fa_blocks)
        self.sa_blocks = nn.ModuleList(sa_blocks)
        self.ca_blocks = nn.ModuleList(ca_blocks)

        self.scale_by_sigma = config.model.scale_by_sigma
        if self.scale_by_sigma:
            self.sigma_map = TimestepEmbedder(self.cond_dim)
            self.output_layer = FinalLayer(
                self.dim,
                self.cond_dim,
                self.patch_size,  # dummy patch size
                self.out_dim)
        else:
            # just standard linear layer
            self.output_layer = nn.Sequential(
                        nn.LayerNorm(self.dim),
                        nn.Linear(self.dim, self.out_dim,
                                  bias=True))

        self.init_params()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    @torch.no_grad()
    def get_grid(self, nlat, nlon, device):
        # create lat, lon grid
        if self.with_poles:
            lat = torch.linspace(-math.pi / 2, math.pi / 2, nlat).to(device)
        else:
            # assume equiangular grid
            lat_end = (nlat - 1) * (2 * math.pi / nlon) / 2
            lat = torch.linspace(-lat_end, lat_end, nlat).to(device)

        lon = torch.linspace(0, 2 * math.pi - (2 * math.pi / nlon), nlon).to(device)
        latlon = torch.stack(torch.meshgrid(lat, lon), dim=-1)
        return latlon, lat, lon

    def forward(self, u, sigma_t, scalar_params, grid_params):
        # u: (batch_size, ny, nx, c + nl*c), sigma: (batch_size, 1), scalar_params: (batch_size, num_cond), grid_params: (batch_size, nlat, nlon, c)

        batch_size = u.size(0)
        nlat, nlon, = u.size(1), u.size(2)
        nlat_grid = nlat // self.patch_size
        nlon_grid = nlon // self.patch_size
        _, lat, lon = self.get_grid(nlat, nlon, u.device)
        _, lat_grid, lon_grid = self.get_grid(nlat_grid, nlon_grid, u.device)

        # n x n distance matrix
        lat_grid_diff = lat_grid.unsqueeze(0) - lat_grid.unsqueeze(1)
        lon_grid_diff = lon_grid.unsqueeze(0) - lon_grid.unsqueeze(1)

        # patchify u
        u = self.patch_embed(u) # [b, nlat, nlon, c] -> [b, nlat//p, nlon//p, dim]

        # patchify grid_params
        grid_emb = self.constant_embedder(grid_params) # [b, nlat//p, nlon//p, dim]

        # patchify pos embed, lat from 0 to pi, lon from -pi to pi
        sphere_pe = self.pe_embed(lat + math.pi/2, lon - math.pi).expand(batch_size, -1, -1, -1) # [b, nlat, nlon, dim]
        sphere_pe = self.pe2patch(sphere_pe) # [b, nlat//p, nlon//p, dim]

        u = u + sphere_pe   # [b, nlat//p, nlon//p, dim]
        grid_emb = grid_emb + sphere_pe # [b, nlat//p, nlon//p, dim]

        if self.scale_by_sigma:
            c_t = F.silu(self.sigma_map(sigma_t))
            c = c_t
        else:
            c_t = None
            c = 0

        if self.num_cond > 0:
            c += self.cond_map(scalar_params)

        # fa blocks
        for l in range(self.num_fa_blocks):
            u = self.fa_blocks[l](u, lat_grid, lat_grid_diff, lon_grid_diff, c)

        # flatten u after factorized attention
        u = rearrange(u, 'b ny nx c -> b (ny nx) c') # [b, nlat//p * nlon//p, dim]
        grid_emb = rearrange(grid_emb, 'b ny nx c -> b (ny nx) c') # [b, nlat//p * nlon//p, dim]

        # dit blocks
        for l in range(self.num_sa_blocks):
            u = self.sa_blocks[l](u, c)
            if l < self.num_ca_blocks:
                u = self.ca_blocks[l](u, grid_emb)

        if self.scale_by_sigma:
            u = self.output_layer(u, c_t)
        else:
            u = self.output_layer(u)

        u = self.unpatchify(u) # [b, nlat, nlon, c]
        return u

    def init_params(self):
        # zero-out constant embedding
        nn.init.constant_(self.constant_embedder[-2].weight, 0)

        # Zero-out output layers:
        if self.scale_by_sigma:

            nn.init.constant_(self.output_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.output_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.output_layer.linear.weight, 0)
            nn.init.constant_(self.output_layer.linear.bias, 0)

        else:
            nn.init.constant_(self.output_layer[1].weight, 0)
            nn.init.constant_(self.output_layer[1].bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**(dim) * C)
        imgs: (N, (t), (D), H, W, C)
        """
        c = self.out_dim
        h, w = self.grid_x, self.grid_y

        assert h * w == x.shape[1]
        ph, pw = self.patch_size, self.patch_size
        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * ph, w * pw)) # [b, c, nlat, nlon]
        imgs = imgs.permute(0, 2, 3, 1) # [b, nlat, nlon, c]

        return imgs
