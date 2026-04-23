import numpy as np
import torch as th

from guided_diffusion.nn import mean_flat
from continuous_diffusion.continuous_diffusion import (
    GaussianDiffusionNoiseLevel, _extract_into_tensor_float64
)


class Turb3dDiffusion(GaussianDiffusionNoiseLevel):
    """
    An adaptation of the GaussianDiffusionNoiseLevel class, tailored for applying 
    physics constraints during training and sampling.

    :param constraint_name: str, name of the constraint function to load.
    :param kwargs: Additional keyword arguments to create the base diffusion process.
    """

    def __init__(self, constraint_name, **kwargs):
        self.constraint_fn = load_constraint(constraint_name)
        super().__init__(**kwargs)

    def sample_noise_level(self, t, broadcast_shape):
        """
        Sample continuous noise levels for a given number of diffusion steps.

        :param t: the number of diffusion steps (minus 1). Here, 0 means step one.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of t.
        :return: noise level tensors of shape [batch_size, 1, ...] 
                 where the shape has K dims.
        """
        sqrt_alpha_bar = _extract_into_tensor_float64(
            self.sqrt_alphas_cumprod, t, broadcast_shape
        )
        sqrt_alpha_bar_prev = _extract_into_tensor_float64(
            self.sqrt_alphas_cumprod_prev, t, broadcast_shape
        )
        sqrt_alpha_bar_sample = (
            (sqrt_alpha_bar - sqrt_alpha_bar_prev) * th.rand_like(sqrt_alpha_bar)
            + sqrt_alpha_bar_prev
        )
        sqrt_one_minus_alpha_bar_sample = th.sqrt(1.0 - sqrt_alpha_bar_sample**2)
        sqrt_recip_alpha_bar_sample = 1.0 / sqrt_alpha_bar_sample
        sqrt_recipm1_alpha_bar_sample = \
            sqrt_one_minus_alpha_bar_sample / sqrt_alpha_bar_sample
        recip_sqrt_recipm1_alpha_bar_sample = \
            sqrt_alpha_bar_sample / sqrt_one_minus_alpha_bar_sample
        return (
            sqrt_alpha_bar_sample.float(), 
            sqrt_one_minus_alpha_bar_sample.float(), 
            sqrt_recip_alpha_bar_sample.float(), 
            sqrt_recipm1_alpha_bar_sample.float(), 
            recip_sqrt_recipm1_alpha_bar_sample.float()
        )

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses using sampled noise levels for given timesteps. 
        The physics constraint defined by self.constraint_fn is applied to modify 
        the model-predicted epsilon.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        sqrt_alpha_bar_sample, sqrt_one_minus_alpha_bar_sample, \
        sqrt_recip_alpha_bar_sample, sqrt_recipm1_alpha_bar_sample, \
        recip_sqrt_recipm1_alpha_bar_sample = self.sample_noise_level(t, x_start.shape)

        x_t = sqrt_alpha_bar_sample * x_start + sqrt_one_minus_alpha_bar_sample * noise

        terms = {}
        index = (slice(None),) + (0,) * (len(x_start.shape) - 1)
        model_output = model(x_t, sqrt_alpha_bar_sample[index], **model_kwargs)

        if self.constraint_fn is not None:
            # compute correction for predicted x_0 using physics constraint
            pred_xstart = sqrt_recip_alpha_bar_sample * x_t \
                - sqrt_recipm1_alpha_bar_sample * model_output
            delta_pred_xstart = self.constraint_fn(pred_xstart) - pred_xstart

            # adjust model-predicted epsilon to enforce the constraint
            model_output = \
                model_output - recip_sqrt_recipm1_alpha_bar_sample * delta_pred_xstart

        target = noise
        assert model_output.shape == target.shape == x_start.shape
        terms["mse"] = mean_flat((target - model_output) ** 2)
        terms["loss"] = terms["mse"]

        return terms


def load_constraint(constraint_name):
    """
    Returns a constraint function based on the given name.
    """
    if constraint_name == "no":
        return None
    elif constraint_name == "2d_ux":
        def constraint_fn(x):
            x = x - x.mean(dim=(-2, -1), keepdim=True)  # zero mean over the yz-plane
            return x
    elif constraint_name == "2d":
        def constraint_fn(x):
            x = zero_mean(x, 0, (-2, -1))  # zero mean of ux over the yz-plane
            return x
    elif constraint_name == "3d_weak":
        def constraint_fn(x):
            x = zero_mean(x, 0, (-2, -1))  # zero mean of ux over the yz-plane
            x = zero_mean(x, 1, (-3, -1))  # zero mean of uy over the xz-plane
            x = zero_mean(x, 2, (-3, -2))  # zero mean of uz over the xy-plane
            return x
    elif constraint_name == "3d_full":
        return incompressible_projection
    else:
        raise NotImplementedError
    return constraint_fn


def zero_mean(x, comp, axis):
    """
    Apply zero-mean adjustment to the specified velocity component over 
    the specified axes.

    :param x: Tensor of shape [B, C, D, H, W], the velocity field.
    :param comp: int, the index of the velocity component to adjust.
    :param axis: tuple of ints, the axes over which to compute the mean.

    Examples:
    - For any comp and axis=(-3, -2, -1): zero mean over the volume.
    - For comp=0 and axis=(-2, -1): zero mean of ux over the yz-plane, 
    as required by incompressibility and volume-averaged zero mean.
    """
    x_comp = x[:, comp] - x[:, comp].mean(dim=axis, keepdim=True)
    x_comp = x_comp.unsqueeze(1)
    x = x.index_copy(1, th.tensor([comp], device=x.device), x_comp)
    return x


def incompressible_projection(x):
    """
    Project the velocity field onto a divergence-free space in Fourier space 
    and remove the zero mode (mean component).

    :param x: Tensor of shape [B, C, D, H, W], the velocity field.
    :return: Tensor of the same shape after the projection.
    """
    B, C, D, H, W = x.shape
    assert C == 3, "Velocity field must have 3 components."
    assert D == H == W, "Only cubic domains are supported."
    assert W % 2 == 0, "Grid size must be even for rFFT."

    # Normalization factor for velocity components
    velocity_scale = th.tensor([1.0, 1.12, 1.12],  # Specific to Rot64_f64_diffusion.h5
        dtype=x.dtype, device=x.device).view(1, C, 1, 1, 1)

    # Compute wave numbers for Fourier space projection
    kx = th.fft.fftfreq(D, device=x.device).view(1, 1, D, 1, 1) * D
    ky = th.fft.fftfreq(H, device=x.device).view(1, 1, 1, H, 1) * H
    kz = th.fft.rfftfreq(W, device=x.device).view(1, 1, 1, 1, W // 2 + 1) * W

    shape = (1, 1, D, H, W // 2 + 1)
    kx, ky, kz = kx.expand(*shape), ky.expand(*shape), kz.expand(*shape)

    kv = th.cat((kx, ky, kz), dim=1)  # Wave vector
    k2 = kx**2 + ky**2 + kz**2
    k2_avoid_zero = th.where(k2 == 0, th.ones_like(k2), k2)  # Prevent division by zero

    x = x * velocity_scale
    x_spec = th.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")

    # Compute divergence-free projection and remove zero mode
    x_spec = x_spec - th.sum(kv * x_spec, dim=1, keepdim=True) * kv / k2_avoid_zero
    x_spec = th.where(k2 > 1e-6, x_spec, th.zeros_like(x_spec))

    x = th.fft.irfftn(x_spec, dim=(-3, -2, -1), norm="ortho")
    x = x / velocity_scale
    return x
