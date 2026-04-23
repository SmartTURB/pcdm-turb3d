from setuptools import setup

setup(
    name="smartturb-diffusion",
    py_modules=[
        "guided_diffusion",
        "continuous_diffusion",
        "palette_diffusion",
        "turb3d_diffusion",
        "physics_diffusion"
    ],
    install_requires=[
        "blobfile>=1.0.5",
        "tqdm",
        "h5py"
    ],
)
