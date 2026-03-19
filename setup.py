"""
setup.py

Package installation configuration.
"""

from pathlib import Path

from setuptools import find_packages, setup


README_PATH = Path(__file__).resolve().parent / "README.md"


setup(
    name="lorentz-transformer",
    version="1.0.0",
    author="papasop",
    description="Minkowski Geometry in Transformer Attention",
    long_description=README_PATH.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/papasop/k-1",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
    ],
)
