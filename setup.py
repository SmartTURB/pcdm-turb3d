from setuptools import setup

setup(
    name="smartturb-diffusion",
    py_modules=[
        "guided_diffusion",
        "continuous_diffusion",
        "turb3d_diffusion"
    ],
    install_requires=[
        "blobfile>=1.0.5",
        "tqdm",
        "h5py"
    ],
)
