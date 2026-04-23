import numpy as np
import torch as th

from guided_diffusion.nn import mean_flat
from guided_diffusion.gaussian_diffusion import (
    ModelMeanType, ModelVarType, LossType, GaussianDiffusion, _extract_into_tensor
)


class GaussianDiffusionNoiseLevel(GaussianDiffusion):
    """
    This class is an adaptation of the GaussianDiffusion class, tailored for diffusion 
    models that condition on a continuous noise level instead of discrete timesteps.
    The implementation follows the method described in:

    Chen, N. et al. (2020). WaveGrad: Estimating Gradients for Waveform Generation. 
    arXiv preprint arXiv:2009.00713.

    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.model_mean_type == ModelMeanType.EPSILON
        assert self.model_var_type in [
            ModelVarType.FIXED_SMALL, ModelVarType.FIXED_LARGE
        ]
        assert self.loss_type == LossType.MSE
        self.sqrt_alphas_cumprod_prev = np.append(1.0, self.sqrt_alphas_cumprod[:-1])

    def p_mean_variance(self, model, x, t, **kwargs):
        def _wrapped_model(x, t, **kwargs):
            sqrt_alpha_bar = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape)
            index = (slice(None),) + (0,) * (len(x.shape) - 1)
            return model(x, sqrt_alpha_bar[index], **kwargs)
        return super().p_mean_variance(_wrapped_model, x, t, **kwargs)

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
        sqrt_one_minus_alpha_bar_sample = th.sqrt(1.0 - sqrt_alpha_bar_sample**2).float()
        sqrt_alpha_bar_sample = sqrt_alpha_bar_sample.float()
        return sqrt_alpha_bar_sample, sqrt_one_minus_alpha_bar_sample

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses using sampled noise levels for given timesteps.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        sqrt_alpha_bar_sample, sqrt_one_minus_alpha_bar_sample = \
        self.sample_noise_level(t, x_start.shape)
        x_t = sqrt_alpha_bar_sample * x_start + sqrt_one_minus_alpha_bar_sample * noise

        terms = {}
        index = (slice(None),) + (0,) * (len(x_start.shape) - 1)
        model_output = model(x_t, sqrt_alpha_bar_sample[index], **model_kwargs)

        target = noise
        assert model_output.shape == target.shape == x_start.shape
        terms["mse"] = mean_flat((target - model_output) ** 2)
        terms["loss"] = terms["mse"]

        return terms


def _extract_into_tensor_float64(arr, timesteps, broadcast_shape):
    """
    Similar to _extract_into_tensor but returns a tensor of dtype float64.
    """
    res = th.from_numpy(arr).to(device=timesteps.device, dtype=th.float64)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
