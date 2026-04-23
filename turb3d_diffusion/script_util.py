from guided_diffusion.script_util import create_model as create_model_nd
from guided_diffusion import gaussian_diffusion as gd
from continuous_diffusion.script_util import continuous_create_gaussian_diffusion
from .unet import UNetModel
from .turb3d_diffusion import Turb3dDiffusion


def create_model_and_diffusion(
    dims,
    image_size,
    in_channels,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    use_continuous_diffusion,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
):
    model = create_model_3d(
        dims,
        image_size,
        in_channels,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = continuous_create_gaussian_diffusion(
        use_continuous_diffusion=use_continuous_diffusion,
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model_3d(
    dims,
    image_size,
    in_channels,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(in_channels if not learn_sigma else 2*in_channels),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        dims=dims,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def turb3d_diffusion_defaults():
    return dict(
        constraint_name="no",
        diffusion_steps=1000,
        sigma_small=False,
        noise_schedule="linear",
        predict_xstart=False,
        rescale_timesteps=False,
    )


def turb3d_model_and_diffusion_defaults():
    res = dict(
        use_unet3d=False,  # use 3D-supported UNetModel or not
        dims=2,
        image_size=64,
        in_channels=3,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(turb3d_diffusion_defaults())
    return res


def turb3d_create_model_and_diffusion(
    use_unet3d,
    dims,
    image_size,
    in_channels,
    class_cond,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    constraint_name,
    diffusion_steps,
    sigma_small,
    noise_schedule,
    predict_xstart,
    rescale_timesteps,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
):
    create_model = create_model_nd if not use_unet3d else create_model_3d
    model = create_model(
        dims,
        image_size,
        in_channels,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=False,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = create_turb3d_diffusion(
        constraint_name=constraint_name,
        steps=diffusion_steps,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
    )
    return model, diffusion


def create_turb3d_diffusion(
    *,
    constraint_name="no",
    steps=1000,
    sigma_small=False,
    noise_schedule="linear",
    predict_xstart=False,
    rescale_timesteps=False,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    model_mean_type = (
        gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
    )
    model_var_type = (
        gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL
    )
    loss_type = gd.LossType.MSE
    return Turb3dDiffusion(
        constraint_name=constraint_name,
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
