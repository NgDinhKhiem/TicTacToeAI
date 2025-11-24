from setuptools import setup, find_packages

setup(
    name="gomoku_rl",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchrl==0.2.1",
        "scipy",
        "omegaconf",
        "hydra-core",
        "tqdm",
        "wandb",
        "matplotlib",
    ],
    python_requires=">=3.10",
)

