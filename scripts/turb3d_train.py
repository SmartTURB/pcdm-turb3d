"""
Train a diffusion model on 2d/3d rotating turbulence fields. 
Physics constraints are applied during training, specified by 
--constraint_name (options: "no", "2d", or "3d").
"""

import argparse

from guided_diffusion import dist_util, logger
from turb3d_diffusion.turb_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from turb3d_diffusion.script_util import (
    turb3d_model_and_diffusion_defaults as model_and_diffusion_defaults,
    turb3d_create_model_and_diffusion as create_model_and_diffusion,
)
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    assert 0 <= args.depth_size <= args.image_size
    if args.depth_size == 0:
        args.depth_size = None

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        depth_size=args.depth_size,
        class_cond=args.class_cond,
        periodic_shift=args.periodic_shift,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        total_steps=args.total_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        dataset_path="",
        dataset_name="",
        depth_size=0,
        periodic_shift=False,
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
        total_steps=1e6,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
