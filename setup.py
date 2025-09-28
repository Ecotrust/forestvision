from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="forestvision",
    version="0.1.0",
    author="Yankuic Galvan",
    author_email="yankuic@gmail.com",
    description="Machine learning package for forest analysis from remote sensing data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ygalvan/forestvision",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.11",
    install_requires=[
        "earthengine-api",
        "geopandas",
        "kornia",
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "lightning[all]",
        "torchgeo[all]",
        "rasterio",
        "scikit-learn",
        "numpy",
        "pandas",
        "matplotlib",
        "shapely>=2.0",
        "torchmetrics",
        "tqdm",
        "dotenv",
        "retry",
        "requests",
        "rio_cogeo",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
            "sphinx",
        ],
        "notebooks": [
            "jupyter",
            "geemap",
            "folium",
        ],
    },
)
