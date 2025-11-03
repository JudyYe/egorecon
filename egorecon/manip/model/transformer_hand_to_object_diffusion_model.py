import logging
import math
import os
import os.path as osp
from inspect import isfunction

import numpy as np
import pytorch3d.ops as pt3d_ops
import torch
import torch.nn.functional as F
from bps_torch.bps import bps_torch
from bps_torch.tools import sample_sphere_uniform
from einops import rearrange, reduce
from jutils import geom_utils, mesh_utils, plot_utils
from pytorch3d.structures import Meshes
from torch import nn
from tqdm.auto import tqdm

from egorecon.utils.motion_repr import HandWrapper

from ..data.utils import get_norm_stats
# from .guidance_optimizer_jax import (do_guidance_optimization, project,
#                                      se3_to_wxyz_xyz, wxyz_xyz_to_se3)
from .guidance_optimizer_hoi_jax import (do_guidance_optimization, project,
                                     se3_to_wxyz_xyz, wxyz_xyz_to_se3, ndc)
from .transformer_module import Decoder

from . import fncmano_jax
from pathlib import Path


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class TransformerDiffusionModel(nn.Module):
    def __init__(
        self,
        d_input_feats,
        d_feats,
        d_model,
        n_dec_layers,
        n_head,
        d_k,
        d_v,
        max_timesteps,
    ):
        super().__init__()

        self.d_feats = d_feats
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k
        self.d_v = d_v
        self.max_timesteps = max_timesteps

        # Input: BS X D X T
        # Output: BS X T X D'
        self.motion_transformer = Decoder(
            d_feats=d_input_feats,
            d_model=self.d_model,
            n_layers=self.n_dec_layers,
            n_head=self.n_head,
            d_k=self.d_k,
            d_v=self.d_v,
            max_timesteps=self.max_timesteps,
            use_full_attention=True,
        )

        self.linear_out = nn.Linear(self.d_model, self.d_feats)

        # For noise level t embedding
        dim = 64
        learned_sinusoidal_dim = 16
        time_dim = dim * 4

        learned_sinusoidal_cond = False
        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model),
        )

    def forward(self, src, noise_t, condition, padding_mask=None):
        # src: BS X T X D (noisy object trajectory)
        # noise_t: BS (timestep)
        # condition: BS X T X (2*D) (left + right hand poses)
        # padding_mask: BS X 1 X (T+1)

        # Concatenate noisy object trajectory with hand pose condition
        B, T, D = src.shape
        if condition.shape[1] > src.shape[1]:
            padding = torch.zeros(
                condition.shape[0], condition.shape[1] - src.shape[1], src.shape[2]
            ).to(condition.device)
            src = torch.cat((src, padding), dim=-2)

        src = torch.cat((src, condition), dim=-1)  # BS X T X (D_out+D_cond+bps?)

        noise_t_embed = self.time_mlp(noise_t)  # BS X d_model
        noise_t_embed = noise_t_embed[:, None, :]  # BS X 1 X d_model

        bs = src.shape[0]
        num_steps = src.shape[1] + 1

        if padding_mask is None:
            # In training, no need for masking
            padding_mask = (
                torch.ones(bs, 1, num_steps).to(src.device).bool()
            )  # BS X 1 X timesteps

        # Get position vec for position-wise embedding
        pos_vec = torch.arange(num_steps) + 1  # timesteps
        pos_vec = (
            pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1)
        )  # BS X 1 X timesteps

        data_input = src.transpose(1, 2).detach()  # BS X (3*D) X T
        feat_pred, _ = self.motion_transformer(
            data_input, padding_mask, pos_vec, obj_embedding=noise_t_embed
        )

        output = self.linear_out(
            feat_pred[:, 1 : 1 + T]
        )  # BS X T X D (predict object trajectory)

        return output  # predicted noise or x0, same size as target object trajectory


class CondGaussianDiffusion(nn.Module):
    def __init__(
        self,
        opt,
        d_feats,
        condition_dim,
        d_model,
        n_head,
        n_dec_layers,
        d_k,
        d_v,
        max_timesteps,
        out_dim,
        timesteps=1000,
        loss_type="l1",
        objective="pred_noise",
        beta_schedule="cosine",
        p2_loss_weight_gamma=0.0,  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k=1,
        **kwargs,
    ):
        super().__init__()
        self.left_mano_model_jax = fncmano_jax.MANOModel.load(Path("assets/mano"), side="left")
        self.right_mano_model_jax = fncmano_jax.MANOModel.load(Path("assets/mano"), side="right")

        self.use_hand_raw_in_cond = False
        if opt.hand == 'cond':
            self.use_hand_raw_in_cond = True
        self.hand_wrapper = HandWrapper(opt.paths.mano_dir)

        self.hand_sensor_encoder = nn.Sequential(
            nn.Linear(in_features=21 * 3 * 2, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=21 * 3 * 2),
        )
        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=1024 * 3, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=condition_dim),
        )

        # For hand-to-object task:
        # d_feats = output object trajectory dimension (9 or 12)
        # condition_dim = input hand poses dimension (2 * d_feats for left + right hand)
        # condition_dim = 3 * d_feats  # Left hand + Right hand
        d_input_feats = d_feats + condition_dim  # Object trajectory + hand poses
        self.opt = opt

        self.denoise_fn = TransformerDiffusionModel(
            # condition_dim=condition_dim,
            d_input_feats=d_input_feats,
            d_feats=d_feats,
            d_model=d_model,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            n_dec_layers=n_dec_layers,
            max_timesteps=max_timesteps,
        )

        self.objective = objective

        self.seq_len = max_timesteps - 1
        self.out_dim = out_dim

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate p2 reweighting
        register_buffer(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -p2_loss_weight_gamma,
        )

        self.bps_path = osp.join(opt.paths.data_dir, "bps/bps-small.pt")
        self.prep_bps_data()
        self.set_metadata()

    def set_metadata(self, *args, **kwargs):
        meta_file = self.opt.traindata.meta_file
        mean, std = get_norm_stats(meta_file, self.opt, 'target')
        self.register_buffer("mean", torch.FloatTensor(mean.reshape(1, 1, -1)))
        self.register_buffer("std", torch.FloatTensor(std.reshape(1, 1, -1)))

    def normalize_data(self, data):
        return (data - self.mean) / self.std

    def denormalize_data(self, data):
        return data * self.std + self.mean

    def predict_start_from_noise_new(self, x_t, t, noise):
        if self.objective == "pred_noise":
            start = (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        elif self.objective == "pred_x0":
            start = noise
        else:
            raise ValueError(f"unknown objective {self.objective}")
        return start

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, x_cond, padding_mask, clip_denoised, rtn_x0=False):
        model_output = self.denoise_fn(x, t, x_cond, padding_mask)

        if self.objective == "pred_noise":
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.objective == "pred_x0":
            x_start = model_output
        else:
            raise ValueError(f"unknown objective {self.objective}")

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        if rtn_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_start
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self, x, t, x_cond, padding_mask=None, clip_denoised=False, guide=False, newPoints=None, hand_raw=None, rtn_info=False, **kwargs
    ):
        b, *_, device = *x.shape, x.device
        x_cond = self.get_cond(x_cond, x, t, newPoints, hand_raw)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=t,
            x_cond=x_cond,
            padding_mask=padding_mask,
            clip_denoised=clip_denoised,
            rtn_x0=True,
        )

        if guide:
            x_start, info = self.guide_jax(x_start, model_kwargs=kwargs)
            model_mean = self.q_posterior(x_start=x_start, x_t=x, t=t)[0]

            # spatial guidance/classifier guidance
            # model_mean = self.guide(model_mean, t, model_kwargs=kwargs)

        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if rtn_info:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, {'mean': model_mean, 'x0': x_start}
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def guide_jax(self, x_start, model_kwargs, t=None):
        x_start_raw = self.denormalize_data(x_start)
        x_dict = self.decode_dict(x_start_raw)

        # x_0_pred, _ = do_guidance_optimization(
        #     traj=se3_to_wxyz_xyz(x_start),
        #     obs=model_kwargs['obs'],
        #     guidance_mode=self.opt.guide.hint,
        #     phase="inner",
        #     verbose=True,
        #     # verbose=False,
        # )
        # x_0_pred = wxyz_xyz_to_se3(x_0_pred)
        info = {}
        if t is not None and t >= 500:
            print(f"Inner-init guidance at t={t}")
            phase = "inner-init"
            pred_dict = x_dict
        else:
            phase = "inner"

            pred_dict, info = do_guidance_optimization(
                pred_dict=x_dict,
                obs=model_kwargs['obs'],
                left_mano_model=self.left_mano_model_jax,
                right_mano_model=self.right_mano_model_jax,
                guidance_mode=self.opt.guide.hint,
                phase=phase,
                verbose=True,
            )
        x_0_pred = self.encode_dict_to_params(pred_dict)
        x_start_opt = self.normalize_data(x_0_pred)
        return x_start_opt, info


    @torch.no_grad()
    def p_sample_loop(self, shape, x_start, x_cond, padding_mask=None, rtn_x_list=False, **kwargs):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        obs = self.get_obs(kwargs.get('guide', False), kwargs.get('obs', {}), shape)
        kwargs['obs'] = obs
        rtn = {'x_list': [], 'x_0_packed_pred': []}
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            p_rtn = self.p_sample(
                x,
                torch.full((b,), i, device=device, dtype=torch.long),
                x_cond,
                padding_mask=padding_mask,
                rtn_info=rtn_x_list,
                **kwargs,
            )
            if rtn_x_list:
                x = p_rtn[0]
                x_info = p_rtn[1]
                rtn['x_list'].append((x.clone(), i))
                rtn['x_0_packed_pred'].append((x_info['x0'].clone(), i))
            else:
                x = p_rtn

        if self.opt.post_guidance and kwargs.get('guide', False):
            x, info = self.guide_jax(x, model_kwargs=kwargs)

        if rtn_x_list:
            return x, rtn
        return x  # BS X T X D

    def get_obs(self, guide, batch, shape):
        obs = {}
        if guide:
            # x2d = project(batch['newPoints'], batch['target_raw'], batch['wTc'], batch['intr'], ndc=True)
            x2d = project(batch['wTc'], batch['intr'], batch['newPoints'], batch['wTo'], ndc=True)

            num_kp = 5
            kp2d = x2d[:, :, :num_kp]  # (B, T, num_kp, 2)
            kp2d_vis = ((kp2d <= 1) & (kp2d >= -1)).all(dim=-1)  

            # if self.opt.guide.hint == "reproj_cd":
            b, T = shape[:2]
            ind_list = []
            kP = 100
            for t in range(T):
                inds = torch.randperm(x2d.shape[2])[:kP]
                ind_list.append(inds)
            ind_list = torch.stack(ind_list, dim=0).to(x2d.device)  # (T, kP)
            ind_list_exp = ind_list[None, :, :, None].repeat(b, 1, 1, 2)
            x2d = torch.gather(x2d, dim=2, index=ind_list_exp)  # (B, T, Q, 2) --? (B, T, kP, 2)
            x2d_vis = ((x2d <= 1) & (x2d >= -1)).all(dim=-1).all(dim=-1)  # (B, T,)

            j2d = project(batch['wTc'], batch['intr'], None, None, wPoints=batch['hand_raw'].reshape(b, T, -1, 3), ndc=ndc)
            j2d_vis = ((j2d <= 1) & (j2d >= -1)).all(dim=-1)  # (B, T, J_2)

            obs = {
                "newPoints": batch['newPoints'],
                "wTc": se3_to_wxyz_xyz(batch['wTc']),
                "intr": batch['intr'],
                "x2d": x2d,
                "kp3d": batch["newPoints"][:, :num_kp],
                "kp2d": kp2d,
                "kp2d_vis": kp2d_vis,
                "x2d_vis": x2d_vis.reshape(b, T, 1),
                "contact": batch['contact'],
                "j3d": batch['hand_raw'].reshape(b, T, -1, 3),
                "j2d": j2d.reshape(b, T, -1, 2),
                "j2d_vis": j2d_vis.reshape(b, T, -1),
            }
            # obs = {
            #     "contact": batch['contact'],
            #     "x": se3_to_wxyz_xyz(batch['target_raw']),
            #     "newPoints": batch['newPoints'],
            #     "wTc": se3_to_wxyz_xyz(batch['wTc']),
            #     "intr": batch['intr'],
            #     "x2d": x2d,
            #     "joints_traj": batch["hand_raw"]
            # }
        return obs

    @torch.no_grad()
    def sample_raw(
        self,
        x_start,
        hand_condition,
        padding_mask=None,
        newPoints=None,
        hand_raw=None,
        rtn_x_list=False,
        **kwargs,
    ):
        rtn = self.sample(
            x_start,
            hand_condition,
            padding_mask=padding_mask,
            newPoints=newPoints,
            hand_raw=hand_raw,
            rtn_x_list=rtn_x_list,
            **kwargs,
        )

        if rtn_x_list:
            motion = rtn[0]
        else:
            motion = rtn

        motion_raw = self.denormalize_data(motion)
        info = {"motion": motion_raw}
        if rtn_x_list:
            info.update(rtn[1])
        return motion_raw, info
        # return motion_raw, {"motion": motion}

    @torch.no_grad()
    def sample(
        self,
        x_start,
        hand_condition,
        cond_mask=None,
        padding_mask=None,
        guide=False,
        newPoints=None,
        hand_raw=None,
        **kwargs,
    ):
        """
        Sample object trajectory conditioned on hand poses.

        Args:
            x_start: BS X T X D - initial object trajectory (can be noise)
            hand_condition: BS X T X (2*D) - left and right hand trajectories
            cond_mask: optional mask for conditional generation
            padding_mask: BS X 1 X (T+1) - mask for variable sequence lengths
        """
        # self.denoise_fn.eval()
        self.eval()

        sample_mode = self.opt.get('sample', 'ddim')
        if sample_mode == "ddim":
            sample_loop_fn = self.ddim_sample_loop
        elif sample_mode == 'ddim_long':
            sample_loop_fn = self.ddim_sample_long
        else:
            sample_loop_fn = self.p_sample_loop
        sample_res = sample_loop_fn(
            x_start.shape,
            x_start,
            hand_condition,
            padding_mask,
            guide=guide,
            newPoints=newPoints,
            hand_raw=hand_raw,
            **kwargs,
        )
        # BS X T X D

        # self.denoise_fn.train()
        self.train()

        return sample_res

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def p_losses(self, x_start, x_cond, t, noise=None, padding_mask=None):
        """
        Compute diffusion losses.

        Args:
            x_start: BS X T X D - ground truth object trajectory
            x_cond: BS X T X (2*D) - hand pose condition (left + right)
            t: BS - noise timesteps
            noise: BS X T X D - optional noise (will be generated if None)
            padding_mask: BS X 1 X (T+1) - optional padding mask
        """
        b, timesteps, d_input = x_start.shape  # BS X T X D
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # noisy object trajectory

        model_out = self.denoise_fn(x, t, x_cond, padding_mask)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        else:
            raise ValueError(f"unknown objective {self.objective}")

        if padding_mask is not None:
            loss = (
                self.loss_fn(model_out, target, reduction="none")
                # * padding_mask[:, 0, 1:][:, :, None]
            )
        else:
            loss = self.loss_fn(model_out, target, reduction="none")  # BS X T X D

        loss = reduce(loss, "b ... -> b (...)", "mean")  # BS X (T*D)

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        loss = loss.reshape(b, timesteps, d_input)
        obj_loss, hand_loss = loss.split([9, d_input - 9], dim=-1)
        hand_loss = self.opt.loss.w_hand * hand_loss
        obj_loss = self.opt.loss.w_wTo * obj_loss
        obj_loss[..., 3:] *= self.opt.loss.w_rot
        loss = torch.cat([obj_loss, hand_loss], dim=-1)


        x0_pred = self.predict_start_from_noise_new(x, t, model_out)
        return loss.mean(), {'model_out': model_out, 'x0_pred': x0_pred}

    def forward(
        self, x_start, hand_condition, padding_mask=None, newPoints=None, hand_raw=None, gt_contact=None,
        training_info={},
    ):
        """
        Forward pass for training.

        Args:
            x_start: BS X T X D - ground truth object trajectory
            hand_condition: BS X T X (2*D) - hand pose condition (left + right)
            cond_mask: optional conditioning mask
            padding_mask: BS X 1 X (T+1) - optional padding mask
        """
        if training_info["step"] < self.opt.train.warmup or self.opt.hand == 'cond':
            self.use_hand_raw_in_cond = True

        bs = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()

        noise = torch.randn_like(x_start)
        xt = self.q_sample(x_start=x_start, t=t, noise=noise)

        cond = self.get_cond(hand_condition, xt, t, newPoints, hand_raw)
        curr_loss, info = self.p_losses(
            x_start, cond, t, noise=noise, padding_mask=padding_mask
        )

        loss = 0
        loss += curr_loss.mean()
        losses = {'diffusion': curr_loss.mean()}

        x0_pred = info['x0_pred']
        B, T, _ = x0_pred.shape
        x0_pred_raw = self.denormalize_data(x0_pred)
        pred_dict = self.decode_dict(x0_pred_raw)

        # add other losses
        # consistency 
        is_warmup = training_info.get('step', 0) < self.opt.train.warmup
        if not is_warmup and self.opt.hand_rep == "joint_theta" and self.opt.loss.w_consistency > 0:
            _, (left_joints_fk, right_joints_fk) = self.decode_hand_mesh(
                pred_dict['left_hand_params'], pred_dict['right_hand_params'], hand_rep='theta', rtn_joints=True)
            left_joints, right_joints = pred_dict['left_joints'], pred_dict['right_joints']

            const_loss = (left_joints_fk.reshape(B, T, -1, 3) - left_joints.reshape(B, T, -1, 3)).norm(dim=-1).mean()
            const_loss += (right_joints_fk.reshape(B, T, -1, 3) - right_joints.reshape(B, T, -1, 3)).norm(dim=-1).mean()
            loss += self.opt.loss.w_consistency * const_loss
            losses['consistency'] = const_loss

        if not is_warmup and self.opt.loss.w_contact > 0:
            wTo_pred = pred_dict['wTo']
            wTo_pred = geom_utils.se3_to_matrix_v2(wTo_pred)  # (B, T, 4, 4)
            wTo_pred = wTo_pred.reshape(B*T, 4, 4)
            oPoints_exp = newPoints[None].repeat(1, T, 1, 1).reshape(B*T, -1, 3)
            wPoints = mesh_utils.apply_transform(oPoints_exp, wTo_pred)  # (B*T, P, 3)

            if self.use_hand_raw_in_cond:
                wHands = hand_raw.reshape(B*T, -1, 3) # (B*T, J, 3)
            else:
                wHands = torch.cat([pred_dict['left_joints'], pred_dict['right_joints']], dim=-1)
                wHands = wHands.reshape(B*T, -1, 3) # (B*T, J, 3)
            J_2 = wHands.shape[-2]

            dist, idx, wNn = pt3d_ops.knn_points(wHands, wPoints, return_nn=True)  # (B*T, 2*J, 3)
            min_dist = dist.reshape(B, T, 2, J_2 // 2)
            pred_contact = (pred_dict["contact"] > 0.5).float()  # (B, T, 2)
            contact_loss = min_dist * pred_contact[..., None]
            contact_loss = contact_loss.mean()

            loss += self.opt.loss.w_contact * contact_loss
            losses['contact'] = contact_loss

        if not is_warmup and self.opt.loss.w_static > 0:
            wPoints = wPoints.reshape(B, T, -1, 3) # (B, T, P, 3)
            wPoints_next = wPoints[:, 1:]
            wPoints_cur = wPoints[:, :-1]
            wPoints_diff = wPoints_next - wPoints_cur  # (B, T-1, P, 3)
            static_loss = (wPoints_diff**2).norm(dim=-1).mean()  # (B, T-1)

            static_mask = (pred_contact[:, 1:] < 0.5) & (pred_contact[:, :-1] < 0.5) 
            static_mask = static_mask[..., 0] & static_mask[..., 1]  # (B, T-1)
            static_loss = static_loss * static_mask.float()  
            static_loss = static_loss.mean()
            loss += self.opt.loss.w_static * static_loss
            losses['static'] = static_loss

        if not is_warmup and self.opt.loss.w_rel_contact > 0:
            assert self.opt.loss.w_contact > 0, "w_rel_contact requires w_contact"
            # relative contact
            # wNn_next: where is the nearest point on the object moves to in the next step
            # print("TODO careful", idx.shape, wPoints.shape)
            wNn_next = torch.gather(wPoints.reshape(B, T, -1, 3)[:, 1:], 2, idx.reshape(B, T, -1, 1)[:, :-1])  # (B, T-1, J, 3)

            p_near = (wNn.reshape(B * T, J_2, 3) - wHands).reshape(B, T, J_2, 3)[:, :-1]
            p_near_next = (wNn_next - wHands.reshape(B, T, J_2, 3)[:, 1:]).reshape(B, T-1, J_2, 3)

            current_rot = wTo_pred.reshape(B, T, 4, 4)[:, :-1, :3, :3].clone()
            next_rot = wTo_pred.reshape(B, T, 4, 4)[:, 1:, :3, :3].clone()

            rot_T_rot = next_rot @ current_rot.transpose(-1, -2)
            res = p_near_next -  p_near @ rot_T_rot.transpose(-1, -2)
            res = res.reshape(B, T-1, J_2, 3)

            dist_prox = torch.norm(res, dim=-1) # (B, T-1, J)

            contact_mask = ((gt_contact[:, :-1] > 0) & (gt_contact[:, 1:] > 0)).float()
            rel_contact_loss = dist_prox.reshape(B, T-1, 2, J_2//2) * contact_mask[..., None]
            rel_contact_loss = rel_contact_loss.mean()

            loss += self.opt.loss.w_rel_contact * rel_contact_loss
            losses['rel_contact'] = rel_contact_loss

        # temporal smoothness: no acceleration
        if not is_warmup and self.opt.loss.w_smoothness > 0:
            x0_pred = info['x0_pred']
            B, T, _ = x0_pred.shape
            x0_pred_raw = self.denormalize_data(x0_pred)
            wTo_pred = self.decode_dict(x0_pred_raw)['wTo']
            
            wTo_pred = geom_utils.se3_to_matrix_v2(wTo_pred)  # (B, T, 4, 4)
            wTo_pred = wTo_pred.reshape(B*T, 4, 4)
            oPoints_exp = newPoints[None].repeat(1, T, 1, 1).reshape(B*T, -1, 3)
            wPoints = mesh_utils.apply_transform(oPoints_exp, wTo_pred)  # (B*T, P, 3)  

            wVel = wPoints[:, 1:] - wPoints[:, :-1]
            wAcc = wVel[:, 1:] - wVel[:, :-1]

            acc_loss = (wAcc**2).mean()
            loss += self.opt.loss.w_smoothness * acc_loss
            losses['smoothness'] = acc_loss


        losses['Total'] = loss
        return loss, losses

    def _get_bps_cond(self, orig_condition, wJoints_depr, xt, newPoints, t):
        if self.opt.condition.bps == 2:
            denorm_x = self.denormalize_data(xt)
            out = self.decode_dict(denorm_x)
            if self.use_hand_raw_in_cond:
                wJoints = wJoints_depr
            else:
                wJoints = torch.cat([out['left_joints'], out['right_joints']], dim=-1)

            wTo_pred = out["wTo"]

            hand = self.encode_hand_sensor_feature(
                wJoints, wTo_pred, newPoints
            )  # (B, T, J*3*2) # per step hand sensor feature
            B = orig_condition.shape[0]
            mask = t <= self.opt.bps_per_t_start
            hand = hand * mask.reshape(B, 1, 1)

            bps_cond = self.encode_bps_feature(newPoints) 
            cond = torch.cat([orig_condition, hand], dim=-1)  # concat in feature dimension
            cond = torch.cat([cond, bps_cond], dim=-2)    # concat in time dimension
            # assert False, "well something is wrong"
        elif self.opt.condition.bps == 1:
            bps_cond = self.encode_bps_feature(newPoints)
            cond = torch.cat([orig_condition, bps_cond], dim=-2)  # (B, T, (2*D) + 1024*3)

        else:
            bs = orig_condition.shape[0]
            bps_cond = torch.zeros([bs, 1, orig_condition.shape[-1]]).to(
                orig_condition.device
            )
            cond = torch.cat([orig_condition, bps_cond], dim=-2)  # (B, T, (2*D))

        return cond
        

    def get_cond(self, hand_condition, xt, t, newPoints, wHands):        
        cond = self._get_bps_cond(hand_condition, wHands, xt, newPoints, t)
        return cond


    def encode_bps(self, newPoints):
        newCom = newPoints.mean(dim=1)  # (1, 3)
        obj_verts = newPoints
        obj_trans = newCom
        obj_bps = self.obj_bps.to(newPoints)
        bps_object_geo = self.bps_torch.encode(
            x=obj_verts,
            feature_type=["deltas"],
            custom_basis=obj_bps.repeat(obj_trans.shape[0], 1, 1)
            + obj_trans[:, None, :],
        )["deltas"]  # T X N X 3
        if self.opt.get("legacy_bps", True):
            logging.warning("Using legacy BPS, ignore newCom")
            return bps_object_geo
        return bps_object_geo + newCom[:, None, :]
        # return bps_object_geo

    def encode_bps_feature(self, newPoints):
        bs = newPoints.shape[0]
        bps_object_geo = self.encode_bps(newPoints)
        bps_feature = self.bps_encoder(bps_object_geo.reshape(-1, 1024 * 3)).reshape(
            bs, 1, -1
        )
        return bps_feature

    def encode_hand_sensor(self, wHands, wTo, oPoints):
        """

        :param wHands: (B, T, J, 3)
        :param wTo: (B, T, 3+6)
        :param oPoints: (B, P, 3)
        :return: (B, T, J*3)
        """
        B, T, J_3 = wHands.shape
        J = J_3 // 3

        P = oPoints.shape[1]
        wTo = geom_utils.se3_to_matrix_v2(wTo)  # (B, T, 4, 4)
        wTo = wTo.reshape(B * T, 4, 4)
        oPoints = oPoints[:, None].repeat(1, T, 1, 1).reshape(B * T, P, 3)
        wPoints = mesh_utils.apply_transform(oPoints, wTo)  # (B*T, P, 3)

        dist, idx, wNn = pt3d_ops.knn_points(
            wHands.reshape(B * T, J, 3), wPoints, return_nn=True
        )
        wNn = wNn.reshape(B, T, J, 3)

        coord = wNn - wHands.reshape(B, T, J, 3)
        return coord.reshape(B, T, J * 3)

    def encode_hand_sensor_feature(self, wHands, wTo, oPoints):
        #
        bs = wHands.shape[0]
        T = wHands.shape[1]
        wCoord = self.encode_hand_sensor(wHands, wTo, oPoints)  # this is delta from wHands
        hand_sensor_feature = self.hand_sensor_encoder(wCoord)

        return hand_sensor_feature

    def prep_bps_data(self):
        n_obj = 1024
        r_obj = 0.25
        if not os.path.exists(self.bps_path):
            bps_obj = sample_sphere_uniform(n_points=n_obj, radius=r_obj).reshape(
                1, -1, 3
            )

            bps = {
                "obj": bps_obj.cpu(),
            }
            print("Generate new bps data to:{0}".format(self.bps_path))
            os.makedirs(osp.dirname(self.bps_path), exist_ok=True)
            torch.save(bps, self.bps_path)

        self.bps = torch.load(self.bps_path)
        self.bps_torch = bps_torch()
        self.register_buffer("obj_bps", self.bps["obj"])

    def ddim_sample_long(self, shape, x_start, x_cond, padding_mask=None, guide=False, rtn_x_list=False, newPoints=None, hand_raw=None, **kwargs):
        print("TODO: first frame ")
        device = x_cond.device
        window_size = self.opt.model.window
        overlap_size = window_size // 2
        seq_len = x_cond.shape[1]
        start_time = None
        canonical_overlap_weights = (
            torch.from_numpy(
                np.minimum(
                    # Make this shape /```\
                    overlap_size,
                    np.minimum(
                        # Make this shape: /
                        np.arange(1, seq_len + 1),
                        # Make this shape: \
                        np.arange(1, seq_len + 1)[::-1],
                    ),
                )
                / overlap_size,
            )
            .to(device)
            .to(torch.float32)
        )
        alpha_bar_t = torch.cat([torch.ones((1, ), device=device), self.alphas_cumprod], dim=0)
        alpha_t = 1 - self.betas

        b = shape[0]
        x_t_packed = torch.randn(shape, device=device)

        obs = self.get_obs(guide=guide, batch=kwargs.get('obs', {}), shape=shape)
        kwargs['obs'] = obs
        ts = quadratic_ts()
        # x_list = []
        rtn = {'x_list': [], 'x_0_packed_pred': [], 'info_inner': [], 'info_post': None}
        for i in tqdm(range(len(ts) - 1)):
            print(f"Sampling {i}/{len(ts) - 1}")
            t = ts[i]
            t_next = ts[i + 1]

            with torch.inference_mode():
                # Chop everything into windows.
                x_0_packed_pred = torch.zeros_like(x_t_packed)
                overlap_weights = torch.zeros((1, seq_len, 1), device=x_t_packed.device)
                
                # Store previous window's model prediction for conditioning
                prev_model_x0_pred = None

                # Denoise each window.
                for start_t in range(0, seq_len, window_size - overlap_size):
                    end_t = min(start_t + window_size, seq_len)
                    assert end_t - start_t > 0

                    overlap_weights_slice = canonical_overlap_weights[
                        None, : end_t - start_t, None
                    ]
                    overlap_weights[:, start_t:end_t, :] += overlap_weights_slice

                    t_pt = torch.full((b,), t, device=device, dtype=torch.long)

                    cond_cur = x_cond[:, start_t:end_t] 
                    # if self.opt.condition.get("first_wTo", False) and start_t > 0 and prev_model_x0_pred is not None:
                    #     # use the prediction from previous window to condition the first step of this sliding window
                    #     # Previous window's last frame before overlap is at local index overlap_size - 1
                    #     ind = window_size - overlap_size
                    #     cond_cur[:, 0, -9:] = prev_model_x0_pred[:, ind, -9:]

                    c = self.get_cond(cond_cur, x_t_packed[:, start_t:end_t], t_pt, newPoints, hand_raw[:, start_t:end_t])
                    model_output = self.denoise_fn(x_t_packed[:, start_t:end_t], t_pt, c, None)
                    model_x0_pred = self.predict_start_from_noise_new(x_t_packed[:, start_t:end_t], t=t, noise=model_output)
                    x_0_packed_pred[:, start_t:end_t] += model_x0_pred * overlap_weights_slice
                    
                    # Store current window's prediction for next iteration
                    prev_model_x0_pred = model_x0_pred


            x_0_packed_pred = x_0_packed_pred / overlap_weights

            sigma_t = torch.cat(
                [
                    torch.zeros((1,), device=device),
                    torch.sqrt(
                        (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t)
                    )
                    * 0.8,
                ]
            )
            if guide:
                x_0_packed_pred, info = self.guide_jax(x_0_packed_pred, model_kwargs=kwargs, t=t)
                rtn['info_inner'].append(info)
            x_t_packed = (
                torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred
                + (
                    torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
                    * (x_t_packed - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred)
                    / torch.sqrt(1 - alpha_bar_t[t] + 1e-1)
                )
                + sigma_t[t] * torch.randn(x_0_packed_pred.shape, device=device)
            )
            if rtn_x_list:
                rtn['x_list'].append((x_t_packed.clone(), t))
                rtn['x_0_packed_pred'].append((x_0_packed_pred.clone(), t))


        if self.opt.post_guidance and kwargs.get('guide', False):
            x_0_packed_pred, info = self.guide_jax(x_0_packed_pred, model_kwargs=kwargs)
            rtn['info_post'].append(info)
        if rtn_x_list:
            return x_t_packed, rtn
        return x_t_packed




    def ddim_sample_loop(
        self,
        shape,
        x_start, 
        x_cond,
        padding_mask=None,
        guide=False,
        rtn_x_list=False,
        newPoints=None,
        hand_raw=None,
        **kwargs,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        device = x_cond.device
        alpha_bar_t = torch.cat([torch.ones((1, ), device=device), self.alphas_cumprod], dim=0)
        alpha_t = 1 - self.betas

        b = shape[0]
        x_t_packed = torch.randn(shape, device=device)

        obs = self.get_obs(guide=guide, batch=kwargs.get('obs', {}), shape=shape)
        kwargs['obs'] = obs
        ts = quadratic_ts()
        # x_list = []
        rtn = {'x_list': [], 'x_0_packed_pred': [], 'info_inner': [], 'info_post': None}
        for i in tqdm(range(len(ts) - 1)):
            print(f"Sampling {i}/{len(ts) - 1}")
            t = ts[i]
            t_next = ts[i + 1]

            t_pt = torch.full((b,), t, device=device, dtype=torch.long)
            # denoise
            c = self.get_cond(x_cond, x_t_packed, t_pt, newPoints, hand_raw)
            model_output = self.denoise_fn(x_t_packed, t_pt, c, padding_mask,)
            x_0_packed_pred = self.predict_start_from_noise_new(x_t_packed, t=t, noise=model_output)

            sigma_t = torch.cat(
                [
                    torch.zeros((1,), device=device),
                    torch.sqrt(
                        (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t)
                    )
                    * 0.8,
                ]
            )
            if guide:
                x_0_packed_pred, info = self.guide_jax(x_0_packed_pred, model_kwargs=kwargs, t=t)
                rtn['info_inner'].append(info)
            x_t_packed = (
                torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred
                + (
                    torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
                    * (x_t_packed - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred)
                    / torch.sqrt(1 - alpha_bar_t[t] + 1e-1)
                )
                + sigma_t[t] * torch.randn(x_0_packed_pred.shape, device=device)
            )
            if rtn_x_list:
                rtn['x_list'].append((x_t_packed.clone(), t))
                rtn['x_0_packed_pred'].append((x_0_packed_pred.clone(), t))


        if self.opt.post_guidance and kwargs.get('guide', False):
            x_0_packed_pred, info = self.guide_jax(x_0_packed_pred, model_kwargs=kwargs)
            rtn['info_post'].append(info)
        if rtn_x_list:
            return x_t_packed, rtn
        return x_t_packed

    def encode_dict_to_params(self, pred_dict):
        wTo = pred_dict['wTo']
        B, T, _ = wTo.shape
        left_hand_params = pred_dict['left_hand_params']
        right_hand_params = pred_dict['right_hand_params']
        contact = pred_dict['contact']
        
        left_joints = pred_dict.get('left_joints', None)
        right_joints = pred_dict.get('right_joints', None)
        
        if self.opt.hand_rep == "joint_theta":
            if left_joints is None:
                _, _, left_joints = self.hand_wrapper.hand_para2verts_faces_joints(left_hand_params, side='left')
                left_joints = left_joints.reshape(B, T, 21*3)
            if right_joints is None:
                _, _, right_joints = self.hand_wrapper.hand_para2verts_faces_joints(right_hand_params, side='right')
                right_joints = right_joints.reshape(B, T, 21*3)

            left_params = torch.cat([left_joints, left_hand_params], dim=-1)
            right_params = torch.cat([right_joints, right_hand_params], dim=-1)
            hand_rep = torch.cat([left_params, right_params], dim=-1)

        x = [wTo]
        hand_io = self.opt.get("hand", "cond")
        if hand_io in ["out", "cond_out"]:
            x.append(hand_rep)
        if self.opt.output.contact:
            x.append(contact)
        x = torch.cat(x, dim=-1)
        return x
        
    def decode_dict(self, target_raw):
        """
        Decode target_raw back into its components.
        
        Args:
            target_raw: [B, T, D] - denormalized target containing:
                - object trajectory (9D: 3D translation + 6D rotation)
                - hand representation (if hand == "out")
                - contact information (if output.contact == True)
        
        Returns:
            dict: Dictionary containing decoded components
        """
        B, T, D = target_raw.shape
        # Start with object trajectory (always first 9D)
        obj_dim = 9
        wTo = target_raw[..., :obj_dim]  # [B, T, 9]
        
        # Track current position in the concatenated tensor
        current_pos = obj_dim
        
        # Extract hand representation if hand == "out"
        left_hand_params = None
        right_hand_params = None
        hand_io = self.opt.get("hand", "cond")
        hand_rep = self.opt.get("hand_rep", "joints")
        if hand_io in ["out", "cond_out"]:
            if hand_rep == "joint":
                # Hand joints: 21 joints * 3D * 2 hands = 126D
                hand_dim = 21 * 3 * 2
                hand_rep = target_raw[..., current_pos:current_pos + hand_dim]  # [B, T, 126]
                left_joints, right_joints = torch.split(hand_rep, 21 * 3, dim=-1)  # [B, T, 63] each
                
                # Convert joints to hand parameters (this would need the inverse of joint2verts_faces_joints)
                # For now, we'll need to implement this conversion or use a different approach
                left_hand_params = left_joints
                right_hand_params = right_joints
                
            elif hand_rep == "theta":
                # Hand theta: (3+3+15+10) * 2 = 62D
                hand_dim = (3 + 3 + 15 + 10) * 2
                hand_rep = target_raw[..., current_pos:current_pos + hand_dim]  # [B, T, 62]
                left_hand_params, right_hand_params = torch.split(hand_rep, hand_dim // 2, dim=-1)  # [B, T, 31] each

            elif hand_rep == "joint_theta":
                hand_dim = 21 * 3 * 2 + (3 + 3 + 15 + 10) * 2
                hand_rep_over = target_raw[..., current_pos:current_pos + hand_dim]  # [B, T, 184]
                left_hand_params_over, right_hand_params_over = torch.split(hand_rep_over, [hand_dim // 2, hand_dim // 2], dim=-1)  # [B, T, 92] each
                left_joints, left_hand_params = torch.split(left_hand_params_over, [21*3, (3+3+15+10)], dim=-1)
                right_joints, right_hand_params = torch.split(right_hand_params_over, [21*3, (3+3+15+10)], dim=-1)
            current_pos += hand_dim
        else:
            left_joints = None
            right_joints = None
            left_hand_params = None
            right_hand_params = None
        
        # Extract contact information if present
        contact = None

        if "output" in self.opt and self.opt.output.contact:
            contact_dim = 2  # left and right hand contact
            contact = target_raw[..., current_pos:current_pos + contact_dim]  # [B, T, 2]
            current_pos += contact_dim
        else:
            contact = torch.zeros([B, T, 2], device=target_raw.device)
        
        rtn = {
            'wTo': wTo,
            'left_hand_params': left_hand_params,
            'right_hand_params': right_hand_params,
            'left_joints': left_joints,
            'right_joints': right_joints,
            'contact': contact,
        }
        return rtn

    def decode_hand_joints(self, left_hand, right_hand):
        device = left_hand.device if left_hand is not None else right_hand.device
        hand_rep = self.opt.hand_rep
        if left_hand is not None:
            if hand_rep == "joint":
                left_joints = left_hand
            elif hand_rep in ["theta", ]:
                _, _, left_joints = self.hand_wrapper.hand_para2verts_faces_joints(left_hand, side='left')
            elif hand_rep == "joint_theta":
                left_joints = left_hand[..., :21*3]
        if right_hand is not None:
            if hand_rep == "joint":
                right_joints = right_hand
            elif hand_rep in ["theta", ]:
                _, _, right_joints = self.hand_wrapper.hand_para2verts_faces_joints(right_hand, side='right')
            elif hand_rep == "joint_theta":
                right_joints = right_hand[..., :21*3]

        B, T, J, _ = left_joints.shape
        left_joints = left_joints.reshape(B, T, J * 3)
        right_joints = right_joints.reshape(B, T, J * 3)
        joints = torch.cat([left_joints, right_joints], dim=-1)
        return joints

    def decode_hand_mesh(self, left_hand, right_hand, hand_rep=None, rtn_joints=False):
        device = left_hand.device if left_hand is not None else right_hand.device
        pref_shape = left_hand.shape[:-1] # 
        flat_dim = int(np.prod(pref_shape))  # product of pref_shape
        D = left_hand.shape[-1]
        
        if hand_rep is None:
            hand_rep = self.opt.hand_rep
        if left_hand is not None:
            if hand_rep == "joint":
                assert len(pref_shape) == 1
                left_hand_meshes = plot_utils.pc_to_cubic_meshes(left_hand.reshape(flat_dim, D // 3, 3))
            elif hand_rep == "theta":
                verts, faces, left_joints = self.hand_wrapper.hand_para2verts_faces_joints(left_hand.reshape(flat_dim, D), side='left')
                left_hand_meshes = Meshes(verts=verts, faces=faces).to(device)
                left_joints = left_joints.reshape(*pref_shape, 21*3)
            elif hand_rep == "joint_theta":
                left_theta = left_hand[..., 21*3:]
                verts, faces, left_joints = self.hand_wrapper.hand_para2verts_faces_joints(left_theta.reshape(flat_dim, D), side='left')
                left_hand_meshes = Meshes(verts=verts, faces=faces).to(device)
                left_joints = left_joints.reshape(*pref_shape, 21*3)
        else:
            left_hand_meshes = None

        if right_hand is not None:
            if hand_rep == "joint":
                assert len(pref_shape) == 1
                right_hand_meshes = plot_utils.pc_to_cubic_meshes(right_hand.reshape(flat_dim, D // 3, 3))
            elif hand_rep == "theta":
                verts, faces, right_joints = self.hand_wrapper.hand_para2verts_faces_joints(right_hand.reshape(flat_dim, D), side='right')
                right_joints = right_joints.reshape(*pref_shape, 21*3)
                right_hand_meshes = Meshes(verts=verts, faces=faces).to(device)
            elif hand_rep == "joint_theta":
                right_theta = right_hand[..., 21*3:]
                verts, faces, right_joints = self.hand_wrapper.hand_para2verts_faces_joints(right_theta.reshape(flat_dim, D), side='right')
                right_joints = right_joints.reshape(*pref_shape, 21*3)
                right_hand_meshes = Meshes(verts=verts, faces=faces).to(device)
        else:
            right_hand_meshes = None

        if rtn_joints:
            return (left_hand_meshes, right_hand_meshes), (left_joints, right_joints)
        return (left_hand_meshes, right_hand_meshes)


def quadratic_ts() -> np.ndarray:
    """DDIM sampling schedule."""
    end_step = 0
    start_step = 1000
    x = np.arange(end_step, int(np.sqrt(start_step))) ** 2
    x[-1] = start_step
    return x[::-1]
