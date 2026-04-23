"""
Print the model summary of UNetModel being used.
"""

import argparse
from torchsummary import summary

from guided_diffusion import dist_util, logger
from continuous_diffusion.script_util import (
    continuous_model_and_diffusion_defaults as model_and_diffusion_defaults,
    create_model_and_diffusion as create_model_and_diffusion_nd,
)
from turb3d_diffusion.script_util import (
    create_model_and_diffusion as create_model_and_diffusion_3d,
)
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()
    assert 0 <= args.depth_size <= args.image_size
    if args.depth_size == 0:
        args.depth_size = None

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    create_model_and_diffusion = (
        create_model_and_diffusion_nd if not args.use_unet3d
        else create_model_and_diffusion_3d
    )
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    if args.depth_size is None:
        summary(model, [(args.in_channels, *(args.dims*[args.image_size])), ()])
    else:
        assert args.dims == 3
        summary(model, [
            (args.in_channels, args.depth_size, args.image_size, args.image_size), ()
        ])


def create_argparser():
    defaults = dict(
        dataset_path="",
        dataset_name="",
        depth_size=0,
        use_unet3d=False,  # use 3D-supported UNetModel or not
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
