"""
setup.py

Package installation configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="lorentz-transformer",
    version="1.0.0",
    author="papasop",
    author_email="your.email@example.com",
    description="Minkowski Geometry in Transformer Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/papasop/k-1",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "black>=22.0", "flake8>=4.0"],
    },
)
