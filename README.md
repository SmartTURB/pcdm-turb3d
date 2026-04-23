# pcdm-turb3d

This is the codebase for [Physics-Constrained Diffusion Model for Synthesis of 3D Turbulent Data](https://arxiv.org/abs/2603.12834).

This repository builds upon [SmartTURB/diffusion-lagr](https://github.com/SmartTURB/diffusion-lagr) and extends it to the generation of 3D turbulent velocity fields with physics-constrained diffusion models.

Specifically, the following additional modules are introduced:

- **[continuous_diffusion](./continuous_diffusion)**: Enables diffusion models to operate on continuous noise levels instead of discrete timesteps. See [WaveGrad](https://arxiv.org/abs/2009.00713) for details.

- **[turb3d_diffusion](./turb3d_diffusion)**: Provides data loading utilities and physics-constrained diffusion pipelines for 3D turbulent velocity fields.

## Installation

<details open>
<summary><strong>Using <code>venv</code> + <code>pip</code> (recommended)</strong></summary>

```bash
python3 -m venv pcdm-turb3d
source pcdm-turb3d/bin/activate

pip install --upgrade pip setuptools wheel

# Optional: load MPI first on systems that provide it as a module
module load mpi

pip install mpi4py

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

pip install -e .
```
</details>

<details>
<summary><strong>Using <code>conda</code></strong></summary>

```bash
conda create -n pcdm-turb3d python=3.7 mpi4py openmpi
conda activate pcdm-turb3d

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

pip install -e .
```
</details>

## Data

The training datasets used in this work are available from the [Smart-TURB portal](http://smart-turb.roma2.infn.it).

Specifically, the 3D turbulent velocity fields at resolution 64x64x64 can be found under the `TURB-Rot` dataset (path: `Files -> data -> 3d_64cubed_pcdm_train`).

Two normalized versions of the same underlying DNS dataset are used:

- `Rot64_diffusion.h5`: dataset-wise normalization (used for DDPM-prog)
- `Rot64_f64_diffusion.h5`: fixed-range normalization (used for all other models)

An archived version of the dataset is available at https://doi.org/10.15161/oar.it/0tb37-k2367, which also includes pretrained model checkpoints and generated samples used in this work.

### Example: Loading the data

```python
import h5py

with h5py.File('Rot64_f64_diffusion.h5', 'r') as f:
    data_norm = f['train'][:]  # shape: (N, 64, 64, 64, 3), normalized to [-1, 1]
    vmin = f['min'][:]         # shape: (3,), per-component minimum
    vmax = f['max'][:]         # shape: (3,), per-component maximum

# Recover physical values
data = (data_norm + 1) * (vmax - vmin) / 2 + vmin
```

A small toy dataset is provided in `datasets/Rot64_f64_diffusion-demo.h5` for quick testing and debugging.

## Training

Please refer to the parent repository's [Training section](https://github.com/SmartTURB/diffusion-lagr#training) for general information.

For 3D turbulent velocity fields, the main additional options are:

- `--depth_size`: size of the depth dimension in the input tensor `(N, depth_size, 64, 64, 3)`. Use `64` (or leave unspecified) for full 3D fields (64x64x64).
- `--periodic_shift True`: enables periodic data augmentation via random cyclic shifts (along two spatial directions for partial depth and all three for full 3D fields).
- `--constraint_name`: selects the physical constraint used by the model.
- `--total_steps`: total number of training iterations.

### Progressive DDPM training

Training can be performed progressively by increasing `depth_size` in stages (8 -> 16 -> 32 -> 48 -> 64), using `--resume_checkpoint` to continue training from the previous stage.

For example (stage with `depth_size 32`):

```bash
DATA_FLAGS="--dataset_path datasets/Rot64_diffusion.h5 --dataset_name train --depth_size 32 --periodic_shift True"
MODEL_FLAGS="--dims 3 --image_size 64 --in_channels 3 --num_channels 32 --num_res_blocks 2 --attention_resolutions 128 --channel_mult 1,2,4,8"
DIFFUSION_FLAGS="--diffusion_steps 2000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 16 --microbatch 8 --resume_checkpoint path/to/checkpoint_depth16.pt --total_steps 4e6"

mpiexec -n $NUM_GPUS python scripts/turb_train.py $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

### Physics-constrained training

The `--constraint_name` flag supports the following options:

- `no` (default): unconstrained training (`DDPM-std`)
- `3d_full`: full 3D constraint (`PCDM-Fourier`)
- `3d_weak`: weak 3D constraint (`PCDM-Integral`)

For example, the following configuration trains the `PCDM-Fourier` model:

```bash
DATA_FLAGS="--dataset_path datasets/Rot64_f64_diffusion.h5 --dataset_name train --depth_size 64 --periodic_shift True"
MODEL_FLAGS="--dims 3 --image_size 64 --in_channels 3 --num_channels 32 --num_res_blocks 2 --attention_resolutions 128 --channel_mult 1,2,4,8"
DIFFUSION_FLAGS="--constraint_name 3d_full --diffusion_steps 2000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 16"

mpiexec -n $NUM_GPUS python scripts/turb3d_train.py $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

## Sampling

Please refer to the parent repository's [Sampling section](https://github.com/SmartTURB/diffusion-lagr#sampling) for general information.

Make sure that `MODEL_FLAGS` and `DIFFUSION_FLAGS` match those used during training. In addition, `--depth_size` must be consistent with the trained model.

To generate samples from a trained model:

```bash
SAMPLE_FLAGS="--num_samples 640 --batch_size 16 --model_path path/to/model.pt"
DATA_FLAGS="--depth_size 64"

python scripts/turb3d_sample.py $SAMPLE_FLAGS $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
```

Pretrained model checkpoints are available at the dataset archive (see [Data](#data) section).

For progressive DDPM models, use `scripts/turb_sample.py` instead of `scripts/turb3d_sample.py`.

After sampling, the generated data are saved as `.npz` files with names such as `samples_640x64x64x64x3-seed000.npz`, where the dimensions correspond to `(num_samples, depth, height, width, channels)`.

You can load and recover the physical velocity fields as follows:

```python
import numpy as np
import h5py

# Load normalization constants (from the training dataset)
with h5py.File('Rot64_f64_diffusion.h5', 'r') as f:
    vmin = f['min'][:]  # shape: (3,)
    vmax = f['max'][:]  # shape: (3,)

# Load generated samples (normalized to [-1, 1])
data_norm = np.load('samples_640x64x64x64x3-seed000.npz')['arr_0']

# Recover physical values
data = (data_norm + 1) * (vmax - vmin) / 2 + vmin
```
