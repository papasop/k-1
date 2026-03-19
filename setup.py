from pathlib import Path
import re

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).parent


def read_version() -> str:
    init_path = ROOT / "lorentz_transformer" / "__init__.py"
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        init_path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError("Unable to find package version.")
    return match.group(1)


setup(
    name="lorentz-transformer",
    version=read_version(),
    author="papasop",
    description="Lorentz Transformer based on K=1 information geometry",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["lorentz_transformer*"]),
    include_package_data=True,
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
